"""Finite-pool chemistry, colored channel counts and leaky first passage.

All stochastic rates/counts are conditional. A fixed pool refreshed to the
reference P/E mixture realizes the previous equal-lifetime mean decay, but
does not identify a biochemical recycling mechanism. Paired random numbers
reduce Monte Carlo error, not biological trial noise.
"""
from dataclasses import dataclass,replace
import numpy as np
from . import branch_current_gain as b


@dataclass(frozen=True)
class Noise:
    molecules: int = 10000
    channels: int = 200
    channel_tau_s: float = .002
    max_channel_g_nS: float = .4
    background_sigma_mV_sqrt_s: float = 20.
    current_pA: float = 8.
    threshold_mV: float = 10.
    horizon_s: float = .05
    burn_s: float = .06


def coupled_binomial(rng,n,p0,p1):
    """Two binomial marginals under a shared-uniform monotone coupling."""
    lo=np.minimum(p0,p1);hi=np.maximum(p0,p1)
    shared=rng.binomial(n,lo)
    prob=np.divide(hi-lo,1-lo,out=np.zeros_like(np.asarray(lo,dtype=float)),where=lo<1)
    extra=rng.binomial(n-shared,np.clip(prob,0,1))
    return shared+extra*(p0>p1),shared+extra*(p1>p0)


def pair_update(rng,c0,c1,N,q0,q1,decay):
    """Exact two-state count transition with frozen q in each arm."""
    on=np.minimum(c0,c1);off=N-np.maximum(c0,c1)
    only0=np.maximum(c0-c1,0);only1=np.maximum(c1-c0,0)
    p01a=q0*(1-decay);p01b=q1*(1-decay)
    p11a=decay+p01a;p11b=decay+p01b
    a0,a1=coupled_binomial(rng,on,p11a,p11b)
    b0,b1=coupled_binomial(rng,off,p01a,p01b)
    d0,d1=coupled_binomial(rng,only0,p11a,p01b)
    e0,e1=coupled_binomial(rng,only1,p01a,p11b)
    return a0+b0+d0+e0,a1+b1+d1+e1


def target_conductance(h,y0,p):
    delta=h-b.baseline(y0,p)
    if p.channel_response=='linear_gate':change=p.capacity_nS*delta
    elif p.channel_response=='bounded_amplifier':change=p.gK_reference_nS*np.tanh(p.capacity_nS/p.gK_reference_nS*delta)
    else:raise ValueError('unknown response')
    return p.gK_reference_nS-change


def simulate_pair(y0,dy,p,n=8192,seed=1,noise=Noise(),dt=.0001,molecular=True,channel=True,bridge=True):
    if not 0<=y0<=y0+dy<=1 or p.eta_E>1:raise ValueError('this paired experiment requires positive P shift and eta_E <= 1')
    if p.tau_P_s!=p.tau_E_s:raise ValueError('fixed-pool refresh supports equal P/E lifetimes only')
    if min(noise.molecules,noise.channels)<1 or min(noise.channel_tau_s,p.C_pF,p.gL_nS,dt)<=0:raise ValueError('invalid counts/rates')
    if noise.background_sigma_mV_sqrt_s<0 or noise.threshold_mV<=0 or noise.max_channel_g_nS<=0:raise ValueError('invalid noise/threshold')
    if p.channel_response=='linear_gate' and p.capacity_nS>b.max_capacity(y0,p)*(1+1e-12):raise ValueError('negative K capacity')
    rngm,rngc,rngv=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    steps=round(noise.horizon_s/dt);dt=noise.horizon_s/steps;burn=round(noise.burn_s/dt)
    dm=np.exp(-dt/p.tau_P_s);dc=np.exp(-dt/noise.channel_tau_s)
    dg=0 if p.tau_gate_s==0 else np.exp(-dt/p.tau_gate_s)
    h=np.full((2,n),b.baseline(y0,p));qref=p.gK_reference_nS/noise.max_channel_g_nS
    if not 0<qref<1:raise ValueError('baseline channel occupancy must be interior')
    if molecular:
        c=rngm.binomial(noise.molecules,y0,n);counts=np.array([c,c])
    else:counts=np.full((2,n),y0)
    if channel:
        c=rngc.binomial(noise.channels,qref,n);opened=np.array([c,c])
    else:opened=np.full((2,n),qref)
    v=np.zeros((2,n));first=np.full((2,n),np.inf);times=[];moments=[]
    for j in range(-burn,steps):
        if j==0:
            if molecular:
                a,c=coupled_binomial(rngm,np.full(n,noise.molecules),y0,y0+dy);counts=np.array([a,c])
            else:counts=np.array([np.full(n,y0),np.full(n,y0+dy)])
        oldp=counts/noise.molecules if molecular else counts.copy()
        if molecular:
            a,c=pair_update(rngm,counts[0],counts[1],noise.molecules,y0,y0,dm);counts=np.array([a,c])
            frac=(oldp+counts/noise.molecules)/2
        else:
            counts=y0+(counts-y0)*dm;frac=(oldp+counts)/2
        x=p.A*(frac+p.eta_E*(1-frac));target=b.gate(x)
        hnew=target+(h-target)*dg
        hm=(h+hnew)/2 if p.tau_gate_s>0 else target
        gtarget=target_conductance(hm,y0,p);q=gtarget/noise.max_channel_g_nS
        if np.min(q)<-1e-12 or np.max(q)>1+1e-12:raise ValueError('insufficient channel conductance capacity')
        q=np.clip(q,0,1)
        oldopen=opened/noise.channels if channel else opened.copy()
        if channel:
            a,c=pair_update(rngc,opened[0],opened[1],noise.channels,q[0],q[1],dc);opened=np.array([a,c])
            conductance=noise.max_channel_g_nS*(oldopen+opened/noise.channels)/2
        else:
            opened=q+(opened-q)*dc;conductance=noise.max_channel_g_nS*(oldopen+opened)/2
        h=hnew
        if j<0:continue # common voltage-clamped preconditioning; release at t=0
        total=p.gL_nS+conductance;rate=1000*total/p.C_pF;decay=np.exp(-rate*dt)
        drive=noise.current_pA+(p.gK_reference_nS-conductance)*(p.Vref_mV-p.EK_mV)
        eq=drive/total
        sigma=noise.background_sigma_mV_sqrt_s
        z=rngv.normal(size=n)
        vnew=eq+(v-eq)*decay+sigma*np.sqrt(-np.expm1(-2*rate*dt)/(2*rate))*z
        hit=vnew>=noise.threshold_mV
        if bridge and sigma>0:
            a=np.maximum(noise.threshold_mV-v,0);c=np.maximum(noise.threshold_mV-vnew,0)
            prob=np.exp(-2*a*c/(sigma*sigma*dt))
            # Local Brownian-bridge approximation to OU crossing; timestep audited.
            hit|=rngv.random(n)<prob
        newly=hit&~np.isfinite(first)
        # Bridge-only crossings placed at interval midpoint; endpoint crossings interpolated.
        fraction=np.full((2,n),.5)
        end=vnew>=noise.threshold_mV
        np.divide(noise.threshold_mV-v,vnew-v,out=fraction,where=end&(vnew!=v))
        first[newly]=(j+np.clip(fraction,0,1))[newly]*dt
        v=vnew
        if j%max(1,steps//100)==0 or j==steps-1:
            I=(p.gK_reference_nS-conductance)*(p.Vref_mV-p.EK_mV)
            times.append((j+1)*dt)
            moments.append([*v.mean(1),*v.std(1),*I.mean(1),*I.std(1),*h.mean(1)])
    return dict(first_s=first,time_s=np.array(times),moments=np.array(moments),dt_s=dt)


def distribution(first,horizon=.05):
    hit=np.isfinite(first);n=len(first);prob=hit.mean();rmst=np.minimum(first,horizon)
    return dict(n=n,probability=float(prob),probability_MC_SE=float(np.sqrt(prob*(1-prob)/n)),
        RMST_ms=float(rmst.mean()*1000),RMST_MC_SE_ms=float(rmst.std(ddof=1)/np.sqrt(n)*1000),
        conditional_median_ms=float(np.median(first[hit])*1000) if hit.any() else None,
        censored=int((~hit).sum()))


def contrast(first,horizon=.05):
    hit=np.isfinite(first).astype(float);d=hit[1]-hit[0]
    dr=(np.minimum(first[1],horizon)-np.minimum(first[0],horizon))*1000
    return dict(delta_probability=float(d.mean()),paired_MC_SE=float(d.std(ddof=1)/np.sqrt(len(d))),
        delta_RMST_ms=float(dr.mean()),paired_RMST_MC_SE_ms=float(dr.std(ddof=1)/np.sqrt(len(d))))


def count_traces(N,q,tau,dt,steps,blocks,rng):
    counts=rng.binomial(N,q,blocks);data=np.empty((blocks,steps+1));data[:,0]=counts/N
    d=np.exp(-dt/tau)
    for j in range(steps):
        counts=rng.binomial(counts,d+q*(1-d))+rng.binomial(N-counts,q*(1-d));data[:,j+1]=counts/N
    return data


def fit_count_noise(traces,q,dt,lag):
    # Known imposed q. Independent unit is the trace/block, not a time sample.
    centered=traces-q;variance=float(np.mean(centered**2))
    cov=float(np.mean(centered[:,:-lag]*centered[:,lag:]));rho=cov/variance
    if not 0<rho<1:raise ValueError('correlation unresolved')
    return dict(effective_count=q*(1-q)/variance,tau_s=-lag*dt/np.log(rho),variance=variance,rho=rho)
