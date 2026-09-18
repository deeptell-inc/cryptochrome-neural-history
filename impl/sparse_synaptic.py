"""Conditional synapse-specific probability modulation in a classical circuit.

Synthetic weights and molecular-to-release gain: no native connectome/EEG fit.
The same unweighted probability budget is allocated to different edge subsets.
"""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import expm
from .neural import readiness_proxy,stop_race
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/sparse-synaptic'
N=64;K=8;P0=.25;PERIOD=.01;FINE=.001
ARMS=('baseline','diffuse','random_sparse','targeted','weak','opposite',
      'diffuse_block_top','diffuse_block_random','rescue','diffuse_block_abs')

def network(seed=26091681,homogeneous=False):
    rng=np.random.default_rng(seed)
    w=rng.lognormal(0,1.2,N);w/=w.sum();w*=.45
    w*=np.where(rng.random(N)<.25,-1.,1.)
    target=np.r_[np.zeros(N//2,int),np.ones(N//2,int)];rng.shuffle(target)
    if homogeneous:w=np.ones(N)*.45/N;target=np.zeros(N,int)
    return w,target

def system(route,prep_leak=.6,coupling=1.5):
    if route=='go':return np.array([[-prep_leak,0.],[coupling,-.6]]),np.array([.25,1.4]),np.array([.10,.25]),1,1.8,.203
    if route=='trigger':return np.array([[-2.]]),np.array([8.]),np.array([.6]),0,.30,.003
    if route=='stop':return np.zeros((1,1)),np.array([5.]),np.array([.65]),0,.80,.003
    raise ValueError(route)

def transition(A,drift,sigma,dt):
    d=len(A);aug=np.zeros((d+1,d+1));aug[:d,:d]=A;aug[:d,-1]=drift
    E=expm(aug*dt);F=E[:d,:d];c=E[:d,-1]
    van=np.block([[A,np.diag(sigma**2)],[np.zeros_like(A),-A.T]])
    V=expm(van*dt);Q=V[:d,d:]@F.T;Q=(Q+Q.T)/2
    return F,c,Q

def influence(w,target,route,prep_leak=.6,coupling=1.5):
    """Independent classical impulse calibration, before molecular intervention.

    Rank by 20--200ms mean preparation state, or 100ms detector/stop state.
    No perturbation outcome or test noise is used to select the subset.
    """
    A,*_=system(route,prep_leak,coupling)
    if route=='go':
        gain=(np.exp(-prep_leak*.02)-np.exp(-prep_leak*.20))/(prep_leak*.18)
        return w*(target==0)*gain
    return w*expm(A*.1)[0,0]

def allocations(scores,seed=26091682):
    top=np.argsort(-scores)[:K];opposite=np.argsort(scores)[:K]
    weak=np.argsort(abs(scores),kind='stable')[:K]
    random=np.random.default_rng(seed).choice(N,K,replace=False)
    absolute=np.argsort(-abs(scores))[:K]
    a=np.zeros((len(ARMS),N));a[1]=1/N
    for row,ids in [(2,random),(3,top),(4,weak),(5,opposite)]:a[row,ids]=1/K
    a[6]=a[1];a[6,top]=0;a[7]=a[1];a[7,random]=0;a[8]=a[1]
    a[9]=a[1];a[9,absolute]=0
    return a,dict(top=top.tolist(),random=random.tolist(),weak=weak.tolist(),opposite=opposite.tolist(),absolute=absolute.tolist())

def source_values():
    import pandas as pd
    path=ROOT/'data/dark-basis-resolution/conditional_products_spikes.csv'
    data=pd.read_csv(path);out={}
    for key,model,gamma in [('full', 'full',0.),('fast_oxygen','full',1e8),('population','population_HQ',0.),('reset','scalar',0.)]:
        row=data.query('condition=="upcj2N__full" and model==@model and gamma_oxygen_s==@gamma').iloc[0]
        raw=float(row.delta_yield);out[key]=dict(delta_yield=0. if abs(raw)<1e-10 else raw,raw_delta_yield=raw,model=model,gamma_oxygen_s=gamma,source=str(path.relative_to(ROOT)))
    return out

def simulate(route,delta_yield,gain,n=4096,seed=26091691,architecture=26091681,
             dt=.002,tau=.012,prep_leak=.6,coupling=1.5,homogeneous=False,release_correlation=0.,profile=None):
    if dt not in (.001,.002):raise ValueError('coupled1/2ms grids only')
    A,drift,sigma,readout,horizon,onset=system(route,prep_leak,coupling)
    F,c,_=transition(A,drift,sigma,dt);ff,_,qf=transition(A,drift,sigma,FINE)
    root=np.linalg.cholesky(qf);factor=round(dt/FINE)
    w,target=network(architecture,homogeneous)
    scores=influence(w,target,route,prep_leak,coupling);alloc,ids=allocations(scores,architecture+1)
    destinations=target if route=='go' else np.zeros(N,int)
    W=np.zeros((N,len(A)));W[np.arange(N),destinations]=w
    rng=np.random.default_rng(seed);syn=np.random.default_rng(seed+100000)
    steps=round(horizon/dt);x=np.zeros((len(ARMS),n,len(A)))
    hits=np.full((len(ARMS),n),-1,int)
    traces=np.empty((steps,len(ARMS),n),np.float32) if route=='go' else None
    aggregate=np.zeros((steps,len(ARMS)));delta_release=np.zeros((len(ARMS),n))
    probability_budget=np.zeros(len(ARMS));expected=np.zeros_like(probability_budget)
    deterministic=np.zeros((len(ARMS),len(A)));det_trace=[]
    pmin=P0;pmax=P0
    for k in range(steps):
        t=k*dt
        if k%round(PERIOD/dt)==0:
            pulse=0. if t<onset else delta_yield*gain*np.exp(-(t-onset)/tau)
            if profile is not None and t>=onset:pulse=gain*float(profile(t-onset))
            dp=pulse*alloc;p=P0+dp
            if np.min(p)<0 or np.max(p)>1:raise ValueError('probability saturation violates equal budget')
            pmin=min(pmin,float(p.min()));pmax=max(pmax,float(p.max()))
            u=syn.random((n,N))
            if release_correlation:
                shared=syn.random((n,1));choose=syn.random((n,1))<release_correlation
                u=np.where(choose,shared,u)
            base=u<P0
            for arm in range(len(ARMS)):
                release=u<p[arm]
                x[arm]+=(release.astype(float)-P0)@W
                delta_release[arm]+=(release.astype(int)-base).sum(1)
            probability_budget+=dp.sum(1)
            deterministic+=dp@W
        eta=np.zeros((n,len(A)))
        for sub in range(factor):eta=eta@ff.T+rng.normal(size=(n,len(A)))@root.T
        x=x@F.T+c+eta[None,:,:];deterministic=deterministic@F.T
        aggregate[k]=x[:,:,0].mean(1);det_trace.append(deterministic[:,0].copy())
        crossed=(hits<0)&(x[:,:,readout]>=1);hits[crossed]=k
        if traces is not None:traces[k]=x[:,:,0]
    rt=np.where(hits>=0,(hits+1)*dt,np.inf)
    records=[];proxies=[]
    # Cue-aligned window uses every trial, avoiding movement-alignment selection.
    lo=round((onset+.02)/dt);hi=min(steps,round((onset+.20)/dt))
    endpoint=round(.4/dt)-1 if route=='go' else min(steps-1,round(.15/dt)-1)
    det=np.asarray(det_trace)
    for arm,name in enumerate(ARMS):
        dtime=np.minimum(rt[arm],horizon)-np.minimum(rt[0],horizon)
        record=dict(route=route,arm=name,seed=seed,architecture=architecture,n=n,dt_s=dt,tau_s=tau,
          gain_probability_per_yield=gain,delta_yield=delta_yield,prep_leak_s=prep_leak,prep_to_go_coupling_s=coupling,
          homogeneous=homogeneous,delta_RMST_s=float(dtime.mean()),MC_SE_RMST_s=float(dtime.std(ddof=1)/np.sqrt(n)),
          release_correlation=release_correlation,
          omission=float(np.isinf(rt[arm]).mean()),mean_RMST_s=float(np.minimum(rt[arm],horizon).mean()),
          expected_extra_release_count=float(probability_budget[arm]),observed_extra_release_count=float(delta_release[arm].mean()),
          delta_state_cue_aligned=float((aggregate[lo:hi,arm]-aggregate[lo:hi,0]).mean()),
          analytic_delta_state_cue_aligned=float(det[lo:hi,arm].mean()),probability_min=pmin,probability_max=pmax,native_prediction=False)
        if route=='go':
            per_trial=(traces[lo:hi,arm]-traces[lo:hi,0]).mean(0)
            record['MC_SE_cue_state']=float(per_trial.std(ddof=1)/np.sqrt(n))
            t,proxy,count=readiness_proxy(traces[:,arm],hits[arm],dt,window=.2)
            # Matched baseline-event alignment isolates neural-state change from time selection.
            tb,pb,cb=readiness_proxy(traces[:,arm],hits[0],dt,window=.2)
            length=round(.2/dt);common=(hits[arm]>=length)&(hits[0]>=length)
            own_hits=np.where(common,hits[arm],-1);base_hits=np.where(common,hits[0],-1)
            _,own,cn=readiness_proxy(traces[:,arm],own_hits,dt,window=.2)
            _,fixed,_=readiness_proxy(traces[:,arm],base_hits,dt,window=.2)
            _,baseline,_=readiness_proxy(traces[:,0],base_hits,dt,window=.2)
            record.update(proxy_delta_own_common=float(own[-1]-baseline[-1]),proxy_delta_fixed_common=float(fixed[-1]-baseline[-1]),
              proxy_alignment_component=float(own[-1]-fixed[-1]),proxy_common_trials=cn)
            proxies.append(dict(arm=name,eligible=count,matched_eligible=cb,common_trials=cn,time_s=t.tolist(),own_alignment=proxy.tolist(),baseline_alignment=pb.tolist(),
              common_own=own.tolist(),common_fixed=fixed.tolist(),common_baseline=baseline.tolist()))
        records.append(record)
    return dict(records=records,rt=rt,hits=hits,aggregate=aggregate,analytic=det,proxies=proxies,
                allocation=alloc,weights=w,targets=destinations,scores=scores,subsets=ids,
                reference_horizon=horizon,dt=dt,probability_budget=probability_budget)

def race_records(go,trigger,stop,ssds=(.2,.4,.6)):
    rows=[];g0=go['rt'][0];d0=trigger['rt'][0];s0=stop['rt'][0]
    for ssd in ssds:
        base=stop_race(g0,d0+s0,np.isfinite(d0),ssd,1.8)[0]
        for route,obj in [('go',go),('trigger',trigger),('stop',stop)]:
            for k,arm in enumerate(ARMS):
                g=obj['rt'][k] if route=='go' else g0
                d=obj['rt'][k] if route=='trigger' else d0
                s=obj['rt'][k] if route=='stop' else s0
                response,cancel,unresolved=stop_race(g,d+s,np.isfinite(d),ssd,1.8)
                diff=response.astype(float)-base
                rows.append(dict(route=route,arm=arm,SSD_s=ssd,response_probability=float(response.mean()),delta_response=float(diff.mean()),
                  MC_SE_response=float(diff.std(ddof=1)/np.sqrt(len(diff))),cancel_probability=float(cancel.mean()),
                  unresolved_probability=float(unresolved.mean()),trigger_failure=float(np.isinf(d).mean())))
    return rows
