"""Explicit counterfactuals on the validated common-rotation cycle.
No physiological claim for switched mobility or accelerated chemical rates.
"""
import numpy as np
from scipy import sparse as sp
from scipy.linalg import eig,null_space,solve
from .hq_rotation import Cycle,LinearSolve,generator,apply_local,s

class CaseCycle(Cycle):
    def __init__(self,f,cutoff=2,tau_s=21.476595751657293e-9):
        super().__init__(f,cutoff,tau_s)
        self.free_generators=dict(self.generators);self.original_tau=tau_s
        self.base_solves=dict(self.solves);self.cache={};self.total_mean_time=0.
    def configure(self,label,multiplier=1.,hq_only=False,protected=()):
        self.label=label;self.rates={'H':10.*multiplier,'R':10. if hq_only else 10.*multiplier,'E':4. if hq_only else 4.*multiplier}
        self.taus={st:np.inf if st in protected else self.original_tau for st in ('H','R','E')}
        for st,rate in self.rates.items():
            key=(st,rate,self.taus[st])
            if key not in self.cache:
                g=self.free_generators[st] if st not in protected else generator(self.f,self.angular,st,np.inf,self.field)
                if rate=={'H':10.,'R':10.,'E':4.}[st] and st not in protected:
                    solver=self.base_solves[st]
                else:
                    d=len(self.f['body'][st]);t=s.trvec(d)
                    a=rate*sp.eye(g.shape[0],format='csc')-g+sp.kron(sp.eye(self.na),sp.csr_matrix(np.outer(t,t)/d*1e8),format='csc')
                    solver=LinearSolve(a,f'{label}:{st}')
                self.cache[key]=(g,solver)
            self.generators[st],self.solves[st]=self.cache[key]
        self.total_mean_time=0.;self.cycle_yields=[]
    def wait(self,state,x,rate):
        d=round(np.sqrt(x.shape[1]));identity=s.trvec(d)/d
        trace=np.einsum('i,aik->ak',s.trvec(d),x)
        xt=x-identity[None,:,None]*trace[:,None,:]
        yt=self.solves[state](rate*xt.reshape(self.na*x.shape[1],-1)).reshape(x.shape)
        yt-=identity[None,:,None]*np.einsum('i,aik->ak',s.trvec(d),yt)[:,None,:]
        dr=0. if not np.isfinite(self.taus[state]) else 1/(6*self.taus[state])
        factors=np.array([rate/(rate+dr*l*(l+1)) for l,m in self.angular['lm']])
        scalar=identity[None,:,None]*(factors[:,None]*trace)[:,None,:]
        self.wait_parts[state]=(scalar,yt)
        return scalar+yt
    def step(self,x):
        p,e=self.react(x);q=float(np.einsum('i,i->',s.trvec(9),p[0,:,0]).real)
        self.cycle_yields.append(q)
        self.total_mean_time+=(1-q)/4e6+q/self.rates['R']+(1-q)/self.rates['E']+1/self.rates['R']+1/self.rates['H']
        returned=self.wait('R',p,self.rates['R'])+apply_local(s.TRACE_E,self.wait('E',e,self.rates['E']))
        reduced=self.wait('R',returned,self.rates['R']);out=self.wait('H',reduced,self.rates['H'])
        self.stages=dict(HQ_wait_traceless_norm=float(np.linalg.norm(self.wait_parts['H'][1][:,:,0])))
        return self.boundary(out)
    def run_case(self,label,multiplier=1.,hq_only=False,protected=()):
        self.configure(label,multiplier,hq_only,protected);x,result=self.run()
        result.update(label=label,multiplier=multiplier,hq_only=hq_only,protected=list(protected),rates_s=self.rates,tau_ns=self.original_tau*1e9,cutoff=round(np.sqrt(self.na))-1,mean_preparation_time_s=self.total_mean_time,cycle_yields=self.cycle_yields)
        return x,result

def storage_curve(cycle,x,times):
    """Carry the prepared full-minus-reset difference through an extra HQ delay."""
    g=cycle.generators['H'].toarray();ident=np.zeros(len(g),complex);ident[:81]=s.trvec(9)/3
    T=null_space(ident[None,:]);red=np.einsum('ia,ij,jb->ab',T.conj(),g,T,optimize=True)
    vals,V=eig(red);dx=(x[:,:,0]-x[:,:,2]).ravel();dx-=ident*np.vdot(ident,dx)
    coeff=solve(V,np.einsum('ia,i->a',T.conj(),dx))
    states=[]
    for t in times:
        # All projected modes should be decaying in the cases used here.
        assert vals.real.max()<1e-4
        vec=np.einsum('ij,j->i',V,np.exp(vals*t)*coeff)
        states.append(np.einsum('ia,a->i',T,vec).reshape(cycle.na,81))
    block=np.stack(states,axis=-1);ys=cycle.yield_values(block)
    return [dict(delay_s=float(t),delta_yield=float(y.real),imag=float(y.imag)) for t,y in zip(times,ys)]
