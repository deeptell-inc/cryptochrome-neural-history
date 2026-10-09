"""Axially symmetric electric orienting potential, E parallel to laboratory B.

rho(n)=pi(n)*sum_a phi_a(n) x_a, pi=exp(kappa*p.n)/Z.
The weighted weak form of the covariant Smoluchowski operator is
-D_R <(L+ad J)phi_a, (L+ad J)phi_b>_pi. It retains rotation
about the dipole axis and exactly preserves the scalar equilibrium mode.
No spin polarization, Stark shift, or physiological dipole is inserted.
"""
import numpy as np
from scipy import sparse as sp
from scipy.linalg import qr, solve_triangular, eigh_tridiagonal
from scipy.special import sph_harm_y
from .hq_rotation import s, ad, Cycle, LinearSolve


def electric_quadrature(nz,kappa):
    """Positive Gauss quadrature for normalized exp(kappa*z) dz on [-1,1]."""
    z,w=np.polynomial.legendre.leggauss(max(160,8*nz))
    w*=np.exp(kappa*z-abs(kappa));w/=w.sum()
    q=np.ones_like(z);previous=np.zeros_like(z);beta=0.;aa=[];bb=[]
    for j in range(nz):
        alpha=np.dot(w,z*q*q);aa.append(alpha)
        r=(z-alpha)*q-beta*previous
        b=np.sqrt(np.dot(w,r*r))
        if j<nz-1:bb.append(b)
        previous,q=q,r/b;beta=b
    nodes,u=eigh_tridiagonal(aa,bb)
    return nodes,u[0]**2


def electric_basis(cutoff,kappa,axis=2,nz=None):
    if axis not in (0,1,2):raise ValueError('axis must be 0, 1, or 2')
    nz=max(5,cutoff+3) if nz is None else nz;nf=2*cutoff+7
    z,w=electric_quadrature(nz,kappa);ph=2*np.pi*np.arange(nf)/nf
    theta=np.arccos(np.repeat(z,nf));phi=np.tile(ph,nz);weights=np.repeat(w/nf,nf)
    n=np.stack([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)],axis=1)
    # Proper cyclic permutation: local z becomes the selected frozen body axis.
    Q=np.roll(np.eye(3),axis-2,axis=0);n=n@Q.T
    lm=[(l,m) for l in range(cutoff+1) for m in range(l,-l-1,-1)]
    raw=np.column_stack([np.sqrt(4*np.pi)*sph_harm_y(l,m,theta,phi) for l,m in lm])
    _,r=qr(np.sqrt(weights[:,None])*raw,mode="economic")
    phase=np.diag(r)/abs(np.diag(r));r=phase.conj()[:,None]*r
    transform=solve_triangular(r,np.eye(len(lm)))
    mul=lambda a,b: np.einsum("ij,jk->ik",a,b,optimize=False)
    Y=mul(raw,transform);P=Y.conj().T*weights
    local=[sp.block_diag([s.spins(l)[a] for l in range(cutoff+1)]).toarray() for a in range(3)]
    deriv=[mul(mul(raw,sum(Q[a,b]*local[b] for b in range(3))),transform) for a in range(3)]
    dd=sum(mul(v.conj().T,weights[:,None]*v) for v in deriv)
    left=[mul(v.conj().T,weights[:,None]*Y) for v in deriv]
    mult=[mul(P,n[:,a,None]*Y) for a in range(3)]
    def clean(x):
        x=x.copy();x[abs(x)<2e-12]=0;return x
    dd=clean(dd);left=[clean(x) for x in left];mult=[clean(x) for x in mult]
    assert np.max(abs(mul(P,Y)-np.eye(len(lm))))<1e-10
    assert np.max(abs(Y[:,0]-1))<1e-10
    return dict(lm=lm,Y=Y,inverse=P,n=n,weights=weights,N=[sp.csr_matrix(x) for x in mult],
                derivative_gram=dd,derivative_left=left,scalar_rotation=-dd,
                kappa=kappa,axis=axis,transform=transform,derivatives=deriv)


def electric_generator(f,angular,state,tau_s,field_T=50e-6,local_noise=True):
    na=len(angular['lm']);d=len(f['body'][state]);Ia=sp.eye(na,format='csr');Id=sp.eye(d*d,format='csr')
    C=[ad(x) for x in f['spin'][state]]
    rot=sp.kron(sp.csr_matrix(angular['derivative_gram']),Id,format='csr')
    for c,left in zip(C,angular['derivative_left']):
        rot+=sp.kron(sp.csr_matrix(left+left.conj().T),c,format='csr')+sp.kron(Ia,c@c,format='csr')
    g=sp.kron(Ia,-1j*ad(f['body'][state])+sp.csr_matrix(f['D'][state] if local_noise else np.zeros((d*d,d*d))),format='csr')
    g+=sum(-1j*field_T*sp.kron(n,ad(z),format='csr') for n,z in zip(angular['N'],f['Z'][state]))
    if np.isfinite(tau_s):g-=rot/(6*tau_s)
    g.eliminate_zeros();return g.tocsc()


class ElectricCycle(Cycle):
    def __init__(self,f,cutoff,kappa,axis=2,tau_s=21.476595751657293e-9,field_T=50e-6,
                 angular_override=None,generator_override=None):
        from scipy.linalg import eigh,lu_factor
        self.f=f;self.angular=(electric_basis(cutoff,kappa,axis) if angular_override is None else angular_override);self.na=len(self.angular['lm'])
        make_generator=electric_generator if generator_override is None else generator_override
        self.field=field_T;self.tau_s=tau_s;self.solves={};self.generators={};self.wait_parts={};self.scalar={}
        for state,rate in [('H',10.),('R',10.),('E',4.),('RP',0.)]:
            g=make_generator(f,self.angular,state,tau_s,field_T);self.generators[state]=g
            a=rate*sp.eye(g.shape[0],format='csc')-g
            if state=='RP':
                loss=1e7*np.kron(s.PS4,np.eye(9))+4e6*np.eye(36)
                a+=sp.kron(sp.eye(self.na),.5*(sp.kron(sp.eye(36),sp.csr_matrix(loss))+sp.kron(sp.csr_matrix(loss.T),sp.eye(36))),format='csc')
            else:
                d=len(f['body'][state]);t=s.trvec(d)
                a+=sp.kron(sp.eye(self.na),sp.csr_matrix(np.outer(t,t)/d*1e8),format='csc')
                dr=0 if not np.isfinite(tau_s) else 1/(6*tau_s)
                self.scalar[state]=lu_factor(rate*np.eye(self.na)-dr*self.angular['scalar_rotation'])
            self.solves[state]=LinearSolve(a,'electric_'+state)
        self.inj=s.linear_map(lambda x:np.kron(s.TRIPLET,x),9,36)
        bra=np.kron(s.SINGLET[None,:],np.eye(9))
        self.product=s.linear_map(lambda x:1e7*np.einsum('ai,ij,bj->ab',bra,x,bra.conj()),36,9)
        self.escape=s.linear_map(lambda x:4e6*s.escape_state(x),36,18)
        self.projectors=[]
        for n in self.angular['n']:
            _,u=eigh(f['body']['H']+field_T*sum(n[a]*f['Z']['H'][a] for a in range(3)))
            cc=np.column_stack([s.vec(np.outer(u[:,j],u[:,j].conj())) for j in range(9)])
            self.projectors.append(np.einsum("ij,kj->ik",cc,cc.conj()))
        self.projectors=np.array(self.projectors)

    def wait(self,state,x,rate):
        from scipy.linalg import lu_solve
        d=round(np.sqrt(x.shape[1]));t=s.trvec(d);identity=t/d
        tr=np.einsum('i,aik->ak',t,x);xt=x-identity[None,:,None]*tr[:,None,:]
        yt=self.solves[state](rate*xt.reshape(self.na*d*d,-1)).reshape(x.shape)
        yt-=identity[None,:,None]*np.einsum('i,aik->ak',t,yt)[:,None,:]
        scalar=identity[None,:,None]*lu_solve(self.scalar[state],rate*tr)[:,None,:]
        self.wait_parts[state]=(scalar,yt);return scalar+yt
