"""Phenomenological classical pathways and task models; no EEG calibration."""
import numpy as np
from .spin import chemical_filter


def environmental_inputs(rng,steps,trials,dt=.01,tau=.08):
    """Hypothesized environment-to-exchange encoding, unrelated to stop cue."""
    a=np.exp(-dt/tau);v=rng.normal(size=trials);indices=np.empty((steps,trials),int)
    for t in range(steps):
        v=a*v+np.sqrt(1-a*a)*rng.normal(size=trials)
        indices[t]=np.rint((np.tanh(.6*v)+1)*4).astype(int)
    return indices


def accumulate(drive,brownian,drift,leak=.6,threshold=1.,dt=.005):
    """OU first passage; latent process continues for time-aligned diagnostics.

    drive is piecewise constant at dt. brownian contains N(0,1) increments.
    Exact OU update for piecewise constant drive; crossings are sampled on grid.
    """
    if drive.shape!=brownian.shape or dt<=0 or leak<0:
        raise ValueError('invalid dynamics inputs')
    steps,n=drive.shape;x=np.zeros(n);traces=np.empty_like(drive)
    hits=np.full(n,-1,int)
    a=np.exp(-leak*dt)
    drift_step=dt if leak==0 else -np.expm1(-leak*dt)/leak
    noise_step=np.sqrt(dt) if leak==0 else np.sqrt(-np.expm1(-2*leak*dt)/(2*leak))
    for t in range(steps):
        x=a*x+drift_step*(drift+drive[t])+.45*noise_step*brownian[t]
        traces[t]=x
        crossed=(hits<0)&(x>=threshold)
        hits[crossed]=t
    rt=np.where(hits>=0,(hits+1)*dt,np.inf)
    return rt,hits,traces


def readiness_proxy(traces,hits,dt,window=1.):
    """Negative accumulator, aligned to first crossing; not microvolt EEG.

    Trials with <window of pre-hit recording are excluded and counted.
    """
    length=int(round(window/dt));eligible=np.flatnonzero(hits>=length)
    if not len(eligible):
        return np.arange(-length,1)*dt,np.full(length+1,np.nan),0
    aligned=np.array([traces[hits[i]-length:hits[i]+1,i] for i in eligible])
    wave=-aligned.mean(0)
    wave-=wave[:max(1,int(.1/dt))].mean()
    return np.arange(-length,1)*dt,wave,len(eligible)


def stop_race(go_rt,stop_latency,triggered,ssd,horizon):
    """Finite stopping latency with explicit omissions/unresolved trials."""
    finish=ssd+np.asarray(stop_latency)
    response=np.isfinite(go_rt)&(go_rt<=horizon)&(~triggered|(go_rt<finish))
    cancel=triggered&(finish<=horizon)&(finish<=go_rt)
    unresolved=~(response|cancel)
    return response,cancel,unresolved


def pathway_steps(dt=.002,duration=8.):
    """Separate illustrative species-specific routes under a unit step in z.

    z -> native CRY binding/activity is unmeasured; these parameters are not fits.
    ROS/TRPA1 has both fast activation and slower loss of availability.
    """
    t=np.arange(0,duration+dt/2,dt)
    z=((t>=1)&(t<5)).astype(float)
    # More effective CRY inhibition lowers cAMP under fixed dopamine drive.
    camp=chemical_filter(1/(1+z),.15,dt,initial=1.)
    # Inhibited K conductance increases depolarizing drive; sign is conditional.
    kv=chemical_filter(z/(1+z),.02,dt,initial=0.)
    # ROS accumulation -> TRPA1 activation + availability reduction.
    ros=chemical_filter(z,.3,dt,initial=0.)
    act=chemical_filter(ros/(.25+ros),.025,dt,initial=0.)
    avail=chemical_filter(1/(1+5*ros),1.2,dt,initial=1.)
    trpa=act*avail
    # Normalized membrane deviations, not cell-specific conductance fits.
    voltage={
        'D1_Gs_cAMP':chemical_filter(camp-1,.02,dt),
        'Kv_beta':chemical_filter(kv,.02,dt),
        'ROS_TRPA1':chemical_filter(trpa,.02,dt),
    }
    return dict(time=t,source=z,camp=camp,kv=kv,ros=ros,availability=avail,
                trpa_current=trpa,**voltage)


def pathway_signal(z,name,dt,reference):
    """Uncalibrated candidate pathway deviation, before population-level gain.

    Identifies effective CRY activity with z as an explicit unverified assumption.
    Filtering outputs end-of-bin states; caller shifts them to maintain causality.
    """
    if name=='D1':
        baseline=1/(1+reference)
        return chemical_filter(1/(1+z),.15,dt,initial=baseline)-baseline
    if name=='Kv':
        baseline=reference/(1+reference)
        return chemical_filter(z/(1+z),.02,dt,initial=baseline)-baseline
    if name=='ROS':
        ros=chemical_filter(z,.3,dt,initial=reference)
        a0=reference/(.25+reference);h0=1/(1+5*reference)
        a=chemical_filter(ros/(.25+ros),.025,dt,initial=a0)
        h=chemical_filter(1/(1+5*ros),1.2,dt,initial=h0)
        return a*h-a0*h0
    raise ValueError(name)
