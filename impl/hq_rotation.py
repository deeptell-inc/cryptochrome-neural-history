"""Finite-field, common rigid-body rotational diffusion for the frozen CRY cycle.

Angular Galerkin basis on n=R^{-1} z; L_rot=-D_R (L_orb+ad J_total)^2.
Only the pre-existing body-frame local noise is retained. No fitted biological tau.
"""
from pathlib import Path
import json,time,hashlib
import numpy as np
from scipy import sparse as sp
from scipy.linalg import eigh
from scipy.special import sph_harm_y
from scipy.sparse.linalg import splu
from scipy import constants as constants
from . import dark_threshold_source as s
from . import dark_md_coefficients as md

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/population-memory/latest_case.npz'

def ad(a):
    n=len(a);return sp.kron(sp.eye(n),sp.csr_matrix(a),format='csr')-sp.kron(sp.csr_matrix(a.T),sp.eye(n),format='csr')

def dense_dissipator(a):
    n=len(a);aa=np.einsum('ji,jk->ik',a.conj(),a)
    return np.kron(a.conj(),a)-.5*(np.kron(np.eye(n),aa)+np.kron(aa.T,np.eye(n)))

def angular_basis(cutoff,quadrature_extra=3):
    lm=[(l,m) for l in range(cutoff+1) for m in range(l,-l-1,-1)]
    nz=max(4,2*cutoff+quadrature_extra);nf=2*nz+1
    z,w=np.polynomial.legendre.leggauss(nz);phi=2*np.pi*np.arange(nf)/nf
    theta=np.arccos(np.repeat(z,nf));phi=np.tile(phi,nz);weights=np.repeat(w/2/nf,nf)
    n=np.stack([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)],axis=1)
    Y=np.column_stack([np.sqrt(4*np.pi)*sph_harm_y(l,m,theta,phi) for l,m in lm])
    inverse=Y.conj().T*weights
    assert np.max(abs(np.einsum('aq,qb->ab',inverse,Y)-np.eye(len(lm))))<1e-12
    orbital=[sp.block_diag([s.spins(l)[a] for l in range(cutoff+1)],format='csr') for a in range(3)]
    mult=[np.einsum('aq,q,qb->ab',inverse,n[:,a],Y) for a in range(3)]
    for a in mult:a[abs(a)<1e-14]=0
    return dict(lm=lm,L=orbital,N=[sp.csr_matrix(a) for a in mult],Y=Y,inverse=inverse,n=n,weights=weights)

def frozen():
    z=dict(np.load(SOURCE));nuc=np.array([s.NI[0][a]+s.NI[1][a] for a in range(3)])
    gn=.403761*constants.physical_constants['nuclear magneton'][0]/constants.hbar
    we=abs(constants.physical_constants['electron g factor'][0])*constants.physical_constants['Bohr magneton'][0]/constants.hbar
    spin={};zeeman={};H={};D={}
    for state,ne in [('H',0),('R',0),('E',1),('RP',2)]:
        e=2**ne;el=[]
        for a in range(3):
            q=np.zeros((e,e),complex)
            for j in range(ne):
                q+=np.kron(np.eye(2**j),np.kron(s.S[a],np.eye(2**(ne-j-1))))
            el.append(np.kron(q,np.eye(9)))
        spin[state]=[np.kron(np.eye(e),nuc[a])+el[a] for a in range(3)]
        zeeman[state]=[we*el[a]-gn*np.kron(np.eye(e),nuc[a]) for a in range(3)]
        if state!='RP':H[state]=z['H_'+state];D[state]=z['D_'+state]
    H['RP']=np.einsum('anbm,cd->acnbdm',H['E'].reshape(2,9,2,9),np.eye(2)).reshape(36,36)
    H['RP']+=50e-6*we*np.kron(np.kron(np.eye(2),s.S[2]),np.eye(9))
    ev,u=eigh(z['K_E']);ops=md.operators('RP');D['RP']=np.zeros((1296,1296),complex)
    for val,vec in zip(ev,u.T):
        if val>0:D['RP']+=val*dense_dissipator(np.einsum('a,aij->ij',vec,ops))
    body={state:H[state]-50e-6*zeeman[state][2] for state in H}
    return dict(archive=z,H=H,body=body,D=D,spin=spin,Z=zeeman)

def generator(f,angular,state,tau_s,field_T=50e-6,local_noise=True):
    count=len(angular['lm']);d=len(f['body'][state]);Id=sp.eye(d*d,format='csr');Ia=sp.eye(count,format='csr')
    C=[ad(x) for x in f['spin'][state]]
    rot=sum((sp.kron(L,Id,format='csr')+sp.kron(Ia,c,format='csr'))@(sp.kron(L,Id,format='csr')+sp.kron(Ia,c,format='csr')) for L,c in zip(angular['L'],C))
    g=sp.kron(Ia,-1j*ad(f['body'][state])+sp.csr_matrix(f['D'][state] if local_noise else np.zeros((d*d,d*d))),format='csr')
    g+=sum(-1j*field_T*sp.kron(n,ad(z),format='csr') for n,z in zip(angular['N'],f['Z'][state]))
    if np.isfinite(tau_s):g-=rot/(6*tau_s)
    g.eliminate_zeros()
    return g.tocsc()

class LinearSolve:
    def __init__(self,A,name):
        self.name=name;self.scale=max(abs(A.data));self.A=A/self.scale
        t=time.monotonic();self.lu=splu(self.A,permc_spec='COLAMD');self.worst=0
        print(json.dumps(dict(factor=name,dimension=A.shape[0],nnz=A.nnz,lu_nnz=self.lu.nnz,seconds=time.monotonic()-t)),flush=True)
    def __call__(self,rhs):
        b=rhs/self.scale;x=self.lu.solve(b)
        residual=np.linalg.norm(self.A@x-b)/max(np.linalg.norm(b),1e-300)
        self.worst=max(self.worst,float(residual));return x

def apply_local(op,x):
    return np.einsum('ij,ajk->aik',op,x,optimize=False)

class Cycle:
    def __init__(self,f,cutoff,tau_s,field_T=50e-6):
        self.f=f;self.angular=angular_basis(cutoff);self.na=len(self.angular['lm']);self.field=field_T;self.tau_s=tau_s;self.solves={};self.generators={};self.wait_parts={}
        for state,rate in [('H',10.),('R',10.),('E',4.),('RP',0.)]:
            g=generator(f,self.angular,state,tau_s,field_T);self.generators[state]=g
            a=rate*sp.eye(g.shape[0],format='csc')-g
            if state=='RP':
                loss=1e7*np.kron(s.PS4,np.eye(9))+4e6*np.eye(36)
                a+=sp.kron(sp.eye(self.na),.5*(sp.kron(sp.eye(36),sp.csr_matrix(loss))+sp.kron(sp.csr_matrix(loss.T),sp.eye(36))),format='csc')
            if state!='RP':
                d=len(f['body'][state]);trace=s.trvec(d)
                # Exact identity/traceless invariant blocks. Lift only the identity
                # block to remove stiffness-induced numerical contamination.
                a+=sp.kron(sp.eye(self.na),sp.csr_matrix(np.outer(trace,trace)/d*1e8),format='csc')
            self.solves[state]=LinearSolve(a,state)
        self.inj=s.linear_map(lambda x:np.kron(s.TRIPLET,x),9,36)
        bra=np.kron(s.SINGLET[None,:],np.eye(9))
        self.product=s.linear_map(lambda x:1e7*np.einsum('ai,ij,bj->ab',bra,x,bra.conj()),36,9)
        self.escape=s.linear_map(lambda x:4e6*s.escape_state(x),36,18)
        self.projectors=[]
        for n in self.angular['n']:
            h=f['body']['H']+field_T*sum(n[a]*f['Z']['H'][a] for a in range(3));_,u=eigh(h)
            cc=np.column_stack([s.vec(np.outer(u[:,j],u[:,j].conj())) for j in range(9)])
            self.projectors.append(np.einsum('ij,kj->ik',cc,cc.conj()))
        self.projectors=np.array(self.projectors)
    def wait(self,state,x,rate):
        d=round(np.sqrt(x.shape[1]));identity=s.trvec(d)/d
        trace=np.einsum('i,aik->ak',s.trvec(d),x)
        xt=x-identity[None,:,None]*trace[:,None,:]
        yt=self.solves[state](rate*xt.reshape(self.na*x.shape[1],-1)).reshape(x.shape)
        yt-=identity[None,:,None]*np.einsum('i,aik->ak',s.trvec(d),yt)[:,None,:]
        dr=0. if not np.isfinite(self.tau_s) else 1/(6*self.tau_s)
        factors=np.array([rate/(rate+dr*l*(l+1)) for l,m in self.angular['lm']])
        scalar=identity[None,:,None]*(factors[:,None]*trace)[:,None,:]
        self.wait_parts[state]=(scalar,yt)
        return scalar+yt
    def react(self,x):
        inj=apply_local(self.inj,x);v=self.solves['RP'](inj.reshape(self.na*1296,-1)).reshape(inj.shape)
        return apply_local(self.product,v),apply_local(self.escape,v)
    def project(self,x):
        values=np.einsum('qa,aik->qik',self.angular['Y'],x)
        values=np.einsum('qij,qjk->qik',self.projectors,values)
        return np.einsum('aq,qik->aik',self.angular['inverse'],values)
    def boundary(self,x):
        y=x.copy();y[:,:,1:2]=self.project(x[:,:,1:2]);y[:,:,2:3]=apply_local(s.RESET,x[:,:,2:3]);return y
    def step(self,x):
        p,e=self.react(x);returned=self.wait('R',p,10.)+apply_local(s.TRACE_E,self.wait('E',e,4.))
        reduced=self.wait('R',returned,10.);out=self.wait('H',reduced,10.)
        def deviation(z):
            tr=np.einsum('i,ai->a',s.trvec(9),z[:,:,0])
            return float(np.linalg.norm(z[:,:,0]-s.X0[None,:]*tr[:,None]))
        self.stages=dict(product_deviation=deviation(p),after_branch_recovery_deviation=deviation(returned),after_reduction_deviation=deviation(reduced),HQ_wait_traceless_norm=float(np.linalg.norm(self.wait_parts['H'][1][:,:,0])))
        return self.boundary(out)
    def initial(self,aligned=False):
        x=np.zeros((self.na,81,3),complex);x[0]=s.X0[:,None]
        if aligned:
            for i,(l,m) in enumerate(self.angular['lm']):
                if m==0:x[i]=np.sqrt(2*l+1)*s.X0[:,None]
        return x
    def yield_values(self,x):
        p,_=self.react(x);return np.einsum('i,ik->k',s.trvec(9),p[0])
    def run(self,cycles=20,aligned=False):
        x=self.initial(aligned=aligned);history=[]
        for i in range(cycles):
            x=self.step(x)
            history.append(float(np.max(abs(np.einsum('i,ik->k',s.trvec(9),x[0])-1))))
        y=self.yield_values(x)
        # Evaluate the tiny spin-history contribution directly, before adding I/d.
        scalar,offset=self.wait_parts['H'];offset=self.boundary(offset)
        oy=self.yield_values(offset)
        values=np.einsum('qa,aik->qik',self.angular['Y'],x)
        minimum=1.;herm=0.
        for q in values:
            for k in range(3):
                rho=s.unvec(q[:,k],9);herm=max(herm,float(np.max(abs(rho-rho.conj().T))));minimum=min(minimum,float(np.linalg.eigvalsh((rho+rho.conj().T)/2).min()))
        return x,dict(yields={k:float(v.real) for k,v in zip(['full','population','reset'],y)},delta_yield=float((y[0]-y[2]).real),direct_spin_contrast=float((oy[0]-oy[2]).real),hq_offset_norm=float(np.linalg.norm(offset[:,:,0])),full_population_difference=float((y[0]-y[1]).real),max_imag_yield=float(abs(y.imag).max()),trace_error=max(history),minimum_angular_density_eigenvalue=minimum,hermiticity_error=herm,last_cycle_stages=self.stages,solve_residuals={k:v.worst for k,v in self.solves.items()})
