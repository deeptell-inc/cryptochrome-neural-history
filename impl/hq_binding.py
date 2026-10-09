"""Mobile/bound stochastic Liouville cycle with a fast small-angle wobble model.

The mobile angular generator is unchanged from hq_rotation. Bound density lives
at anchor quadrature nodes; capture/release conserves spin and is orientation
selective. Bound wobble is a *reduced*, extreme-narrowing GKSL approximation,
not a fitted neuronal motion law. All exchange is retained during reaction.
"""
import time
import numpy as np
from scipy import sparse as sp
from scipy.linalg import eigh, expm, lu_factor, lu_solve
from scipy.linalg.blas import zgemm
from .hq_rotation import (Cycle, LinearSolve, angular_basis, generator, ad,
                          dense_dissipator, apply_local, s)


def wobble_operators(f, rms_deg, tau_s):
    """Three-axis Gaussian small-angle noise, total RMS angle in degrees.

    <theta_a(t) theta_b(0)> = delta_ab theta_rms^2/3 exp(-|t|/tau).
    D_extra = 2 theta_rms^2 tau/3 sum_a D[-i[J_a,H_body]].
    Six positive-weight rotations approximate the mean H and local bath to
    second order; Zeeman is fixed in the anchor frame and not rotated.
    """
    angle=np.deg2rad(rms_deg); variance=angle**2/3
    out={}; validity={}
    for st in ('H','R','E','RP'):
        h=f['body'][st]; d=len(h); mean_h=np.zeros_like(h,dtype=complex); mean_d=np.zeros_like(f['D'][st],dtype=complex)
        extra=np.zeros_like(mean_d)
        if not angle:
            mean_h=h.copy(); mean_d=f['D'][st].copy()
        else:
            for j in f['spin'][st]:
                v=-1j*(np.einsum('ij,jk->ik',j,h)-np.einsum('ij,jk->ik',h,j))
                extra+=2*variance*tau_s*dense_dissipator(v)
                for sign in (-1,1):
                    u=expm(-1j*sign*angle*j); su=np.kron(u.conj(),u)
                    mean_h+=zgemm(1,zgemm(1,u,h),u.conj().T)/6
                    mean_d+=zgemm(1,zgemm(1,su,f['D'][st]),su.conj().T)/6
        out[st]=dict(body=mean_h,D=mean_d+extra,extra=extra)
        bandwidth=float(np.ptp(eigh(h,eigvals_only=True)))
        validity[st]=dict(body_bandwidth_rad_s=bandwidth,bandwidth_tau=bandwidth*tau_s,
                          trace_defect=float(np.linalg.norm(np.einsum('i,ij->j',s.trvec(d),extra))),
                          unital_defect=float(np.linalg.norm(np.einsum('ij,j->i',extra,s.trvec(d)))))
    return out,validity


class BindingCycle:
    def __init__(self,f,cutoff=2,tau_free=21.476595751657293e-9,
                 rms_deg=0.,tau_w=1e-11,field_T=50e-6,build_reaction=True,mobile_reference=None,
                 quadrature_extra=None,angular_override=None,generator_override=None,node_model=None):
        self.f=f; self.cutoff=cutoff; self.field=field_T; self.tau_free=tau_free
        self.angular=(angular_basis(cutoff,quadrature_extra=2-cutoff if quadrature_extra is None else quadrature_extra)
                      if angular_override is None else angular_override)
        make_generator=generator if generator_override is None else generator_override
        self.na=len(self.angular['lm']); self.nq=len(self.angular['n'])
        self.Y=self.angular['Y']; self.P=self.angular['inverse']; self.w=self.angular['weights']
        self.wobble,self.validity=wobble_operators(f,rms_deg,tau_w)
        self.rms_deg=rms_deg; self.tau_w=tau_w
        self.gf={}; self.gb={}; self.brp=[]; self.afrp=None; self.stats={}
        self.inj=s.linear_map(lambda x:np.kron(s.TRIPLET,x),9,36)
        bra=np.kron(s.SINGLET[None,:],np.eye(9))
        self.product=s.linear_map(lambda x:1e7*np.einsum('ai,ij,bj->ab',bra,x,bra.conj()),36,9)
        self.escape=s.linear_map(lambda x:4e6*s.escape_state(x),36,18)
        loss=1e7*np.kron(s.PS4,np.eye(9))+4e6*np.eye(36)
        lossop=.5*(np.kron(np.eye(36),loss)+np.kron(loss.T,np.eye(36)))
        self.projectors={}
        if mobile_reference is not None:
            assert (cutoff,tau_free,field_T)==(mobile_reference.cutoff,mobile_reference.tau_free,mobile_reference.field)
        for st in ('H','R','E','RP'):
            if st=='RP' and not build_reaction:continue
            self.gf[st]=(mobile_reference.gf[st] if mobile_reference is not None else
                         make_generator(f,self.angular,st,tau_free,field_T))
            b=self.wobble[st]; base=-1j*ad(b['body']).toarray()+b['D']
            zz=[ad(x).toarray() for x in f['Z'][st]]
            if node_model is None:
                self.gb[st]=np.array([base-1j*field_T*sum(n[a]*zz[a] for a in range(3))
                                      for n in self.angular['n']])
            else:
                nodes=[node_model(st,n) for n in self.angular['n']]
                self.gb[st]=np.array([-1j*ad(nb['body']).toarray()+nb['D']
                    -1j*field_T*sum(n[a]*zz[a] for a in range(3))
                    for n,nb in zip(self.angular['n'],nodes)])
                if st=='H':bound_bodies=[nb['body'] for nb in nodes]
            if st=='RP':
                if mobile_reference is None:
                    a=-self.gf[st]+sp.kron(sp.eye(self.na),sp.csr_matrix(lossop),format='csc')
                    self.afrp=LinearSolve(a,'free_RP')
                else:self.afrp=mobile_reference.afrp
                for g in self.gb[st]: self.brp.append(lu_factor(lossop-g))
        for kind,h in [('f',f['body']['H']),('b',self.wobble['H']['body'])]:
            projs=[]
            for q,n in enumerate(self.angular['n']):
                hq=bound_bodies[q] if node_model is not None and kind=='b' else h
                _,u=eigh(hq+field_T*sum(n[a]*f['Z']['H'][a] for a in range(3)))
                c=np.column_stack([s.vec(np.outer(u[:,j],u[:,j].conj())) for j in range(9)])
                projs.append(np.einsum('ij,kj->ik',c,c.conj()))
            self.projectors[kind]=np.array(projs)
        # Keep only the LU factors for bound RP, saving about one factor-set RAM.
        if 'RP' in self.gb:del self.gb['RP']

    def evaluate(self,x): return np.einsum('qa,aik->qik',self.Y,x,optimize=True)
    def project_angular(self,x): return np.einsum('aq,qik->aik',self.P,x,optimize=True)
    def total_trace(self,x):
        a,b=x; d=round(np.sqrt(a.shape[1])); t=s.trvec(d)
        return t@a[0]+np.einsum('q,i,qik->k',self.w,t,b)

    def configure(self,fraction=.5,koff=1.,bias=0.,build_wait=True,profile_override=None):
        assert 0<=fraction<1 and koff>=0 and -.99<=bias<=1.98
        self.fraction=fraction;self.koff=koff;self.bias=bias
        self.kon=fraction/(1-fraction)*koff
        self.profile=1+bias*.5*(3*self.angular['n'][:,2]**2-1)
        if profile_override is not None:
            self.profile=np.asarray(profile_override,float)
            if self.profile.shape!=(self.nq,) or np.any(self.profile<=0) or abs(self.w@self.profile-1)>1e-10:
                raise ValueError('capture profile must be positive and normalized')
        self.capture=self.kon*self.profile
        self.mult=np.einsum('aq,q,qb->ab',self.P,self.capture,self.Y)
        self.mult[abs(self.mult)<1e-13*max(1.,self.capture.max())]=0
        self.wait_solvers={};self.bound_wait={};self.scalar_solvers={}
        self.stats=dict(wait_iterations=0,wait_fixed_point_error=0.,reaction_iterations=0,
                        reaction_fixed_point_error=0.)
        if not build_wait:return
        for st,rate in [('H',10.),('R',10.),('E',4.)]:
            d=len(self.f['body'][st]);t=s.trvec(d);lift=np.outer(t,t)/d*1e8
            a=rate*sp.eye(self.na*d*d,format='csc')-self.gf[st]
            a+=sp.kron(sp.csr_matrix(self.mult),sp.eye(d*d),format='csc')
            a+=sp.kron(sp.eye(self.na),sp.csr_matrix(lift),format='csc')
            self.wait_solvers[st]=LinearSolve(a,'exchange_'+st)
            self.bound_wait[st]=[lu_factor((rate+koff)*np.eye(d*d)-g+lift) for g in self.gb[st]]
            dr=1/(6*self.tau_free)
            scalar=(np.diag([rate+dr*l*(l+1) for l,m in self.angular['lm']])
                    if 'scalar_rotation' not in self.angular else
                    rate*np.eye(self.na)-dr*self.angular['scalar_rotation'])
            scalar+=rate/(rate+koff)*self.mult.real if np.max(abs(self.mult.imag))<1e-12 else 0
            if np.max(abs(self.mult.imag))>=1e-12: scalar=scalar.astype(complex)+rate/(rate+koff)*self.mult
            self.scalar_solvers[st]=lu_factor(scalar)

    @staticmethod
    def clean(x):
        d=round(np.sqrt(x.shape[1]));t=s.trvec(d)
        return x-t[None,:,None]/d*np.einsum('i,aik->ak',t,x)[:,None,:]

    @staticmethod
    def solve_nodes(factors,x):
        return np.array([lu_solve(lu,v) for lu,v in zip(factors,x)])

    def wait(self,st,x,rate):
        xf,xb=x;d=round(np.sqrt(xf.shape[1]));t=s.trvec(d);ident=t/d
        tf=np.einsum('i,aik->ak',t,xf);tb=np.einsum('i,qik->qk',t,xb)
        rhs=rate*tf+self.koff*rate/(rate+self.koff)*(self.P@tb)
        yf_scalar=lu_solve(self.scalar_solvers[st],rhs)
        yb_scalar=(rate*tb+self.capture[:,None]*(self.Y@yf_scalar))/(rate+self.koff)
        rf=rate*self.clean(xf);rb=rate*self.clean(xb)
        def bsolve(v):return self.clean(self.solve_nodes(self.bound_wait[st],v))
        def fsolve(v):return self.clean(self.wait_solvers[st](v.reshape(self.na*d*d,-1)).reshape(v.shape))
        bf=bsolve(rb);fixed=fsolve(rf+self.koff*self.project_angular(bf));yf=fixed
        for it in range(30):
            corr=bsolve(self.capture[:,None,None]*self.evaluate(yf))
            yn=fixed+fsolve(self.koff*self.project_angular(corr))
            err=np.linalg.norm(yn-yf)/max(np.linalg.norm(yn),1e-30);yf=yn
            if err<2e-12:break
        else:raise RuntimeError('wait exchange did not converge')
        self.stats['wait_iterations']=max(self.stats['wait_iterations'],it+1)
        self.stats['wait_fixed_point_error']=max(self.stats['wait_fixed_point_error'],float(err))
        yb=bsolve(rb+self.capture[:,None,None]*self.evaluate(yf))
        return (yf+ident[None,:,None]*yf_scalar[:,None,:],
                yb+ident[None,:,None]*yb_scalar[:,None,:])

    def react(self,x,exchange=True,return_integral=False):
        inj=tuple(apply_local(self.inj,v) for v in x)
        def rsolve(v):
            vf=self.afrp(v[0].reshape(self.na*1296,-1)).reshape(v[0].shape)
            vb=self.solve_nodes(self.brp,v[1]);return vf,vb
        base=rsolve(inj);v=base
        if exchange and self.kon+self.koff:
            for it in range(12):
                jump=self.capture[:,None,None]*self.evaluate(v[0])-self.koff*v[1]
                dv=rsolve((-self.project_angular(jump),jump))
                vn=tuple(a+b for a,b in zip(base,dv))
                err=sum(np.linalg.norm(a-b) for a,b in zip(vn,v))/max(sum(np.linalg.norm(a) for a in vn),1e-30)
                v=vn
                if err<1e-12:break
            else:raise RuntimeError('RP exchange did not converge')
            self.stats['reaction_iterations']=max(self.stats['reaction_iterations'],it+1)
            self.stats['reaction_fixed_point_error']=max(self.stats['reaction_fixed_point_error'],float(err))
        if return_integral:return v
        return tuple(apply_local(self.product,a) for a in v),tuple(apply_local(self.escape,a) for a in v)

    def boundary(self,x):
        a,b=(z.copy() for z in x)
        va=self.evaluate(a[:,:,1:2]);va=np.einsum('qij,qjk->qik',self.projectors['f'],va)
        a[:,:,1:2]=self.project_angular(va)
        b[:,:,1:2]=np.einsum('qij,qjk->qik',self.projectors['b'],b[:,:,1:2])
        a[:,:,2:3]=apply_local(s.RESET,a[:,:,2:3]);b[:,:,2:3]=apply_local(s.RESET,b[:,:,2:3])
        return a,b

    def step(self,x):
        p,e=self.react(x);p=self.wait('R',p,10.);e=self.wait('E',e,4.)
        ret=tuple(a+apply_local(s.TRACE_E,b) for a,b in zip(p,e))
        return self.boundary(self.wait('H',self.wait('R',ret,10.),10.))

    def initial(self):
        a=np.zeros((self.na,81,3),complex);a[0]=(1-self.fraction)*s.X0[:,None]
        b=self.fraction*self.profile[:,None,None]*s.X0[None,:,None]*np.ones((1,1,3))
        return a,b

    def yields(self,x):return self.total_trace(self.react(x)[0])

    def run(self,cycles=20):
        x=self.initial();traceerr=0.;history=[];t0=time.monotonic()
        for _ in range(cycles):
            x=self.step(x);traceerr=max(traceerr,float(max(abs(self.total_trace(x)-1))))
            history.append(float(np.linalg.norm(x[1][:,:,0]-x[1][:,:,2])))
        y=self.yields(x);dx=tuple(v[:,:,:1]-v[:,:,2:3] for v in x);dy=self.yields(dx)[0]
        motion=self.yields(tuple(apply_local(s.RESET,v[:,:,:1]) for v in x))[0]
        herm=0.;mineig=1.
        for values in (self.evaluate(x[0]),x[1]):
            for node in values:
                for k in range(3):
                    rho=s.unvec(node[:,k],9);herm=max(herm,float(np.max(abs(rho-rho.conj().T))))
                    mineig=min(mineig,float(np.linalg.eigvalsh((rho+rho.conj().T)/2).min()))
        out=dict(fraction=self.fraction,koff_s=self.koff,kon_s=self.kon,bias=self.bias,
                 rms_deg=self.rms_deg,tau_w_s=self.tau_w,tau_free_s=self.tau_free,cutoff=self.cutoff,
                 anchor_nodes=self.nq,cycles=cycles,yields={k:float(v.real) for k,v in zip(('full','population','reset'),y)},
                 delta_yield=float(dy.real),subtraction_delta=float((y[0]-y[2]).real),
                 instantaneous_spin_contribution=float((y[0]-motion).real),
                 motion_distribution_contribution=float((motion-y[2]).real),
                 full_population_difference=float((y[0]-y[1]).real),trace_error=traceerr,
                 hermiticity_error=herm,minimum_density_eigenvalue=mineig,
                 max_imaginary_yield=float(max(abs(y.imag))),seconds=time.monotonic()-t0,
                 observed_bound_fractions=np.einsum('q,i,qik->k',self.w,s.trvec(9),x[1]).real.tolist(),
                 bound_history_norm=history,stats=self.stats.copy(),validity=self.validity)
        return x,out
