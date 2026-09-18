"""Conditional finite-chemistry -> stochastic synapses -> pre-motor veto.

A discrete packet model, not a native calibrated neural circuit. Nuclear-reset
control still contains the same single-pair quantum reaction map.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from .sparse_synaptic import network
from .stochastic_branch_spike import coupled_binomial, pair_update

ROOT = Path(__file__).resolve().parents[1]
MODES = ('majority', 'weighted', 'temporal', 'competition')
ALLOCS = ('diffuse', 'random', 'top', 'weak', 'opposite')
ROUTES = ('go', 'trigger', 'stop')

@dataclass(frozen=True)
class Config:
    gain: float = 256.
    k: int = 8
    tau: float = .012
    pool: int = 10000
    p0: float = .45
    rho: float = 0.
    chemical_delay: float = .01
    cue: float = .2
    threshold: float = 1.
    offset: float = 0.
    heterogeneity: float = .15
    latency_scale: float = 1.
    architecture: int = 26091681
    homogeneous: bool = False
    route: str = 'trigger'
    source: str = 'full'
    period: float = .01
    horizon: float = 1.
    blocked: bool = False


def sources():
    df = pd.read_csv(ROOT/'data/dark-basis-resolution/conditional_products_spikes.csv')
    df = df[(df.condition=='upcj2N__full') & (df.gamma_oxygen_s==0)].set_index('model')
    q0 = float(df.loc['full','yield_reference'])
    return q0, {k:float(df.loc[m,'yield_P']) for k,m in
                [('full','full'),('population','population_HQ'),('reset','scalar')]}


def architecture(c, names=ALLOCS):
    w,_ = network(c.architecture)
    if c.homogeneous: w=np.ones(64)
    delay=np.random.default_rng(c.architecture+2).integers(0,4,64)*.01
    score=w*np.exp(-delay/.04)
    ids={'top':np.argsort(-score,kind='stable')[:c.k],
         'weak':np.argsort(abs(score),kind='stable')[:c.k],
         'opposite':np.argsort(score,kind='stable')[:c.k],
         'random':np.random.default_rng(c.architecture+1).choice(64,c.k,False)}
    a=np.zeros((len(names),64))
    for i,name in enumerate(names):
        if name=='diffuse': a[i]=1/64
        else: a[i,ids[name]]=1/c.k
    return w,delay,a,score


def probability(count_fraction, c, alpha, q0):
    cap=.95*min(c.p0,1-c.p0)*c.k
    budget=cap*np.tanh(c.gain*(np.asarray(count_fraction)-q0)/cap)
    return c.p0+budget[...,None]*alpha


def uniforms(rng,n,m,rho):
    independent=rng.random((n,m))
    shared=rng.random((n,1)); select=rng.random((n,1))<rho
    return np.where(select,shared,independent)


def source_trace(c,n,seed,q0,q1,recovery='stationary'):
    """Exact finite-population transition on the observation grid.

    Pulse preparation at cue is a common protocol reset of the chemical pool,
    not a stop-dependent reset. Baseline pulse has stationary q0 distribution.
    """
    rng=np.random.default_rng(seed)
    steps=round(c.horizon/c.period)
    count=rng.binomial(c.pool,q0,n); a=count.copy(); b=count.copy()
    path=np.empty((steps,2,n)); d=np.exp(-c.period/c.tau)
    initial=np.full(n,c.pool)
    for j in range(steps):
        if j==round(c.cue/c.period):
            a,b=coupled_binomial(rng,initial,q0,q1)
        elif j:
            target=0. if recovery=='depletion' and j>round(c.cue/c.period) else q0
            a,b=pair_update(rng,a,b,c.pool,target,target,d)
        path[j]=np.array([a,b])/c.pool
    return path


def classify(times, horizon):
    g,d,s=times[...,0],times[...,1],times[...,2]
    stop=np.isfinite(s)&(s<g)
    go=np.isfinite(g)&~stop
    unresolved=~(stop|go)
    return np.stack([stop,go,unresolved],-1)


def simulate(c=Config(),n=1024,seed=26091701,names=ALLOCS,replay=None,
             detector_enabled=True,recovery='stationary'):
    if not (1<=c.k<=16 and 0<c.p0<1 and c.pool>=1 and c.tau>0 and 0<=c.rho<=1):
        raise ValueError('invalid counts, rates or probabilities')
    if c.route not in ROUTES or c.gain<0 or c.chemical_delay<0 or c.heterogeneity<0:
        raise ValueError('invalid scenario')
    if abs(round(c.cue/c.period)*c.period-c.cue)>1e-9:
        raise ValueError('cue must lie on packet grid')
    q0,ys=sources()
    source=c.source
    q1=ys.get(source,ys['full'])
    if source=='reset': q1=q0 # raw reset difference ~5e-15 is solver tolerance
    elif source not in ('full','population','classical_mean','classical_stochastic','replay'):
        raise ValueError(source)
    if recovery not in ('stationary','depletion'): raise ValueError('recovery')
    if recovery=='depletion' and source=='classical_mean': raise ValueError('mean depletion not implemented')
    path=source_trace(c,n,seed+10,q0,q1,recovery) if replay is None else replay.copy()
    times_grid=np.arange(len(path))*c.period
    if source=='classical_mean':
        mean=q0+(q1-q0)*np.exp(-np.maximum(times_grid-c.cue,0)/c.tau)*(times_grid>=c.cue)
        path[:,0]=q0; path[:,1]=mean[:,None]
    if path.shape!=(round(c.horizon/c.period),2,n): raise ValueError('replay shape')
    w,delay,alpha,score=architecture(c,names)
    L=len(names); shape=(4,L,2,n)
    # score has SD 1 for independent baseline releases, common across architectures.
    sd=np.sqrt(c.p0*(1-c.p0))
    wm=np.ones(64)/(8*sd); ww=w/(np.linalg.norm(w)*sd)
    ws=np.sign(w)/64 # mV per event, explicitly assumed passive soma proxy
    edge_delay=np.rint(delay/c.period).astype(int)
    if np.max(abs(edge_delay*c.period-delay))>1e-10: raise ValueError('edge delay must lie on grid')
    queues=np.zeros((edge_delay.max()+1,L,2,n,3,3)) # route, [vote,weight,soma]
    state=np.zeros(shape+(3,)); hits=np.full(shape+(3,),np.inf)
    soma=np.zeros((L,2,n)); soma_means=[]
    # Stable per-trial excitability variation is a simulated nuisance, not measured biology.
    rng=np.random.default_rng(seed+20)
    offsets=c.offset+rng.normal(0,c.heterogeneity,(n,3))
    release_rng=np.random.default_rng(seed+30)
    decay=np.exp(-c.period/.04); norm=np.sqrt((1-decay)/(1+decay))
    zthr=(32-64*c.p0)/(8*sd)*c.threshold
    budgets=np.zeros((L,2)); input_sum=np.zeros((L,2)); var_sum=np.zeros((L,2))
    cond_var_sum=np.zeros((L,2)); mean_prob_min=1.; mean_prob_max=0.
    exact_budget_error=0.; arrivals=0
    source_after_trigger=np.zeros(shape)
    route=ROUTES.index(c.route)
    steps=len(path); burn=edge_delay.max()
    for j in range(-burn,steps):
        t=j*c.period; slot=j%len(queues)
        if j<0:
            frac=np.full((2,n),q0)
        else: frac=path[j]
        reached=(t>=c.cue+c.chemical_delay-1e-12) and not c.blocked
        # Before chemical arrival, both arms transmit the same stationary chemical noise.
        # Arrival gates intervention only; baseline chemical susceptibility remains active.
        if not reached: frac=np.broadcast_to(frac[0],(2,n))
        p=probability(frac[None,:,:],c,alpha[:,None,None,:],q0)
        mean_prob_min=min(mean_prob_min,float(p.min())); mean_prob_max=max(mean_prob_max,float(p.max()))
        if j>=0:
            diff=p[:,1]-p[:,0]
            exact_budget_error=max(exact_budget_error,float(np.max(abs(diff.sum(-1)-diff.sum(-1)[0]))))
            budgets+=p.sum(-1).mean(-1)-64*c.p0
            mu=(p-c.p0)@ww
            input_sum+=mu.mean(-1)
            var_sum+=mu.var(-1)
            cond_var_sum+=(p*(1-p)*ww**2).sum(-1).mean(-1)
        for rr in range(3):
            u=uniforms(release_rng,n,64,c.rho)
            rel=(u[None,None,:,:]<p) if rr==route else np.broadcast_to(u<c.p0,(L,2,n,64))
            centered=rel-c.p0
            for dl in np.unique(edge_delay):
                mask=edge_delay==dl
                # Arrival time: chemical survival already uses cue age at release.
                coeff=np.stack([wm[mask],ww[mask],ws[mask]],-1)
                scores=centered[...,mask]@coeff
                queues[(slot+dl)%len(queues),:,:,:,rr,:]+=scores
        arrived=queues[slot].copy(); queues[slot]=0
        if j<0: continue
        active=np.ones(shape+(3,),bool)
        active[...,1]=detector_enabled&(t>=c.cue-1e-12)&(t<=c.cue+.25+1e-12)
        active[...,2]=np.isfinite(hits[...,1])&(t>hits[...,1]+1e-12)
        age=np.zeros(shape+(3,)); age[...,0]=t; age[...,1]=t-c.cue
        age[...,2]=np.where(np.isfinite(hits[...,1]),t-hits[...,1],-1)
        evidence=np.stack([arrived[...,0],arrived[...,1],arrived[...,1],arrived[...,1]])
        prev=state.copy()
        filtered=decay*state+(1-decay)*evidence
        state[0]=evidence[0];state[1]=evidence[1];state[2:]=filtered[2:]
        # Simultaneous mutual inhibition, active only after the detector has triggered.
        coupled=active[3,...,2]
        state[3,...,0]-=(1-decay)*.3*np.maximum(prev[3,...,2],0)*coupled
        state[3,...,2]-=(1-decay)*.3*np.maximum(prev[3,...,0],0)*coupled
        state[~active]=0
        read=state.copy();read[2:]/=norm
        read+=offsets
        eligible=active & (age>=np.array([.35,.04,.07])*c.latency_scale-1e-12)
        crossed=eligible & (read>=zthr-1e-12) & ~np.isfinite(hits)
        hits[crossed]=t
        # Detector and stop continue as latent processes after actual motor onset.
        source_after_trigger+=active[...,2]*max(0,t>=c.cue+c.chemical_delay)*np.exp(-max(t-c.cue,0)/c.tau)
        soma=np.exp(-c.period/.02)*soma+arrived[...,route,2]
        soma_means.append(soma.mean(-1))
        arrivals+=int(reached)
    outcomes=classify(hits,c.horizon)
    return dict(config=asdict(c),names=list(names),modes=MODES,times=hits,outcomes=outcomes,
        path=path,weights=w,delays_s=delay,alpha=alpha,scores=score,
        budget=budgets,input_mean=input_sum/steps,chemical_input_var=var_sum/steps,
        independent_release_var=cond_var_sum/steps,soma_means=np.array(soma_means),
        source_exposure_after_trigger=source_after_trigger*c.period,
        probability_range=[mean_prob_min,mean_prob_max],budget_error=exact_budget_error,
        zthreshold=zthr,source_arrival_before_horizon=bool(c.cue+c.chemical_delay<c.horizon),
        reference_yield=q0,source_yield=q1,n=n,seed=seed)


def summarize(result):
    c=result['config']; rows=[]; n=result['n']; times=result['times']; outcomes=result['outcomes']
    for m,mode in enumerate(MODES):
        for a,name in enumerate(result['names']):
            t=times[m,a]; o=outcomes[m,a]
            row=dict(**c,seed=result['seed'],n=n,integration=mode,allocation=name,
                budget_error=result['budget_error'],p_min=result['probability_range'][0],p_max=result['probability_range'][1])
            for oi,label in enumerate(('stop','continue','unresolved')):
                d=(o[1,:,oi].astype(float)-o[0,:,oi])*100
                row.update({f'{label}_ref':float(o[0,:,oi].mean()),f'{label}_source':float(o[1,:,oi].mean()),
                    f'delta_{label}_pp':float(d.mean()),f'{label}_MC_SE_pp':float(d.std(ddof=1)/np.sqrt(n)),
                    f'{label}_paired_disagreements':int(np.count_nonzero(d))})
            for ri,label in enumerate(ROUTES):
                cap=min(c['horizon'],c['cue']+.25) if label=='trigger' else c['horizon']
                d=(np.minimum(t[1,:,ri],cap)-np.minimum(t[0,:,ri],cap))*1000
                row[f'{label}_delta_RMST_ms']=float(d.mean())
                row[f'{label}_RMST_MC_SE_ms']=float(d.std(ddof=1)/np.sqrt(n))
                for ar,an in enumerate(('ref','source')):
                    hit=np.isfinite(t[ar,:,ri]);row[f'{label}_{an}_censor']=float((~hit).mean())
                    row[f'{label}_{an}_median_s']=float(np.median(t[ar,hit,ri])) if hit.any() else None
            for ar,an in enumerate(('ref','source')):
                row[f'trigger_failure_{an}']=float(np.isinf(t[ar,:,1]).mean())
                row[f'trigger_late_{an}']=float((np.isfinite(t[ar,:,1])&(t[ar,:,1]>=t[ar,:,0])).mean())
                row[f'chemical_arrival_after_go_{an}']=float((c['cue']+c['chemical_delay']>=t[ar,:,0]).mean())
                row[f'posttrigger_exposure_{an}_s']=float(result['source_exposure_after_trigger'][m,a,ar].mean())
                row[f'weighted_input_mean_{an}']=float(result['input_mean'][a,ar])
                row[f'chemical_input_variance_{an}']=float(result['chemical_input_var'][a,ar])
                row[f'independent_release_variance_{an}']=float(result['independent_release_var'][a,ar])
            sm=result['soma_means'][:,a,1]-result['soma_means'][:,a,0]
            row['soma_peak_mean_delta_mV_assumed']=float(sm[np.argmax(abs(sm))])
            row['expected_extra_release_count']=float(np.diff(result['budget'][a])[0])
            rows.append(row)
    return rows
