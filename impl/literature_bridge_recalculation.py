"""Recalculate the old exponential-input task with separated Hk chemistry.

All new rates are explicit scenarios from chemical-carrier-mapping. Neither
oxidized occupancy nor normalized response is a quantum or behavioral effect.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from .chemical_carrier_mapping import cofactor_generator

def solve_pulse(a0,off,binding,tau,*,method='DOP853',duration=10.,rtol=2e-11,atol=2e-14):
    if not np.all(np.isfinite([a0,off,binding,tau,duration])) or min(a0,off,binding,tau,duration)<=0:raise ValueError('positive finite parameters')
    def rhs(t,y):
        a=a0*np.exp(-t/tau);G=cofactor_generator(a,off,binding,0.)
        p=y[:3];s=y[3:]
        return np.r_[G@p,G@s+a*np.array([-p[0],p[0],0.])]
    # Tangent s=d p / d log(a0) separates large total writing from a small contrast.
    # Bound explicit solver steps even after a narrow forcing pulse decays.
    # Dense-output tangent differences are otherwise vulnerable to stiff-tail
    # interpolation artifacts when perturbing a0.
    max_step=min(tau/4,.5/max(off,binding,a0)) if method=='DOP853' else np.inf
    sol=solve_ivp(rhs,(0,duration),[1.,0.,0.,0.,0.,0.],method=method,rtol=rtol,atol=atol,dense_output=True,max_step=max_step)
    if not sol.success:raise RuntimeError(sol.message)
    return sol

def peak(sol):
    t=np.r_[0,np.geomspace(1e-7,sol.t[-1],2000)]
    y=sol.sol(t)[1];candidates=[(t[0],y[0]),(t[-1],y[-1])]
    for j in np.flatnonzero((y[1:-1]>=y[:-2])&(y[1:-1]>=y[2:]))+1:
        ans=minimize_scalar(lambda u:-sol.sol(u)[1],bounds=(t[j-1],t[j+1]),method='bounded',options={'xatol':1e-12})
        candidates.append((float(ans.x),float(-ans.fun)))
    return max(candidates,key=lambda z:z[1])

def held_steady_occupancy(a0,off,binding):
    # At steady state: a R=b O=h U, R+O+U=1.
    return (1/off)/(1/a0+1/off+1/binding)

def old_pulse(t,k,tau):
    t=np.asarray(t,float)
    if np.isclose(k,1/tau,rtol=1e-12):return k*t*np.exp(-k*t)
    return k*(np.exp(-k*t)-np.exp(-t/tau))/(1/tau-k)
