"""Product -> saturating current -> exact stochastic threshold -> go/detect/stop.

Perfect integrate-and-fire (zero leak) under a held chemical readout. Continuous
first-passage distributions remove the previous grid-crossing error. All neuron
and circuit coefficients are declared scenarios, not fitted native parameters.
"""
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.special import ndtr,log_ndtr
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.stats import norm
from .dark_threshold_source import OUT,save
from .neural import stop_race

SSDS=(.1,.2,.3,.4,.6)
ROLES=('preparation','go','detection','stop')
DRIFTS=np.array([1.8,15.,5.])
NOISE=np.array([.45,2.,.45])
HORIZON=3.
DETECTION_DEADLINE=.15


def fpt_cdf(t,mu,sigma,boundary=1.):
    if sigma<=0 or boundary<=0:raise ValueError('noise and boundary must be positive')
    t=np.asarray(t,float);safe=np.maximum(t,1e-300)
    v=sigma*np.sqrt(safe)
    with np.errstate(over='ignore',invalid='ignore'):
        result=ndtr((mu*safe-boundary)/v)+np.exp(2*mu*boundary/sigma**2+log_ndtr(-(mu*safe+boundary)/v))
    return np.where(t>0,np.clip(result,0,1),0.)


def fpt_pdf(t,mu,sigma,boundary=1.):
    t=np.asarray(t,float);safe=np.maximum(t,1e-300)
    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
        logf=np.log(boundary/sigma)-.5*np.log(2*np.pi)-1.5*np.log(safe)-(boundary-mu*safe)**2/(2*sigma**2*safe)
    return np.where(t>0,np.exp(logf),0.)


def go_cdf(t,mu,sigma,boundary=1.,initial_variance=0.):
    """Gaussian initial readiness mixture, using a closed-form normal identity.

    Extends the boundary integral to the whole normal distribution. Error is
    bounded by the above-threshold and exponentially tilted tail masses;
    production scenarios enforce that bound below 1e-8 for either drift sign.
    """
    if initial_variance==0:return fpt_cdf(t,mu,sigma,boundary)
    if boundary<=0 or initial_variance<0:raise ValueError('mixture control requires positive mean boundary')
    t=np.asarray(t,float);v=np.sqrt(sigma**2*np.maximum(t,0)+initial_variance)
    logscale=2*mu*boundary/sigma**2+2*mu**2*initial_variance/sigma**4
    result=ndtr((mu*t-boundary)/v)+np.exp(logscale+log_ndtr((-mu*t-boundary-2*mu*initial_variance/sigma**2)/v))
    return np.clip(result,0,1)


def mixture_error_bound(mu,sigma,boundary,variance):
    if variance==0:return 0.
    # Above-boundary normal mass plus an exponentially tilted tail. Valid for
    # either sign of drift; for positive drift this is <= twice the tail mass.
    return float(ndtr(-boundary/np.sqrt(variance))+np.exp(
        2*mu*boundary/sigma**2+2*mu**2*variance/sigma**4+
        log_ndtr((-boundary-2*mu*variance/sigma**2)/np.sqrt(variance))))


@lru_cache(None)
def go_restricted_mean(mu,sigma,boundary,initial_variance,horizon):
    return quad(lambda t:float(1-go_cdf(t,mu,sigma,boundary,initial_variance)),0,horizon,epsabs=1e-10)[0]


@lru_cache(None)
def restricted_mean(mu,sigma,boundary,horizon):
    return quad(lambda t:float(1-fpt_cdf(t,mu,sigma,boundary)),0,horizon,epsabs=1e-10)[0]


@lru_cache(None)
def conditional_median(mu,sigma,boundary,horizon):
    hit=float(fpt_cdf(horizon,mu,sigma,boundary))
    if hit<1e-14:return None
    return brentq(lambda t:float(fpt_cdf(t,mu,sigma,boundary))-.5*hit,1e-14,horizon)


def spike_metrics(product,reference,G_pA,C_pF=20.,I0_pA=4.,margin_mV=10.,noise_mV_sqrt_s=20.,half=.02,window=.05,refractory=.005):
    if min(product,reference)<0 or min(C_pF,margin_mV,half)<=0:raise ValueError('invalid chemical or neuronal state')
    gate=lambda p:p/(half+p)
    current=I0_pA+G_pA*(gate(product)-gate(reference))
    mu=1000*current/C_pF  # pA/pF -> V/s -> mV/s
    rate=max(mu,0)/(margin_mV+refractory*max(mu,0))
    variance_rate=margin_mV*noise_mV_sqrt_s**2/(margin_mV+refractory*mu)**3 if mu>0 else None
    return dict(current_pA=current,delta_current_pA=current-I0_pA,voltage_drift_mV_s=mu,
        firing_probability=float(fpt_cdf(window,mu,noise_mV_sqrt_s,margin_mV)),
        first_spike_RMST_s=restricted_mean(mu,noise_mV_sqrt_s,margin_mV,window),
        first_spike_conditional_median_s=conditional_median(mu,noise_mV_sqrt_s,margin_mV,window),
        renewal_rate_Hz=rate,renewal_count_variance_per_s=variance_rate,first_spike_window_s=window)


@lru_cache(None)
def nodes(n):return np.polynomial.legendre.leggauss(n)


@lru_cache(None)
def race_probabilities(drifts,ssd,n=160,horizon=HORIZON,deadline=DETECTION_DEADLINE,go_boundary=1.,noises=tuple(NOISE),initial_variance=0.):
    """Independent first-passage races with an explicit detector deadline.

    Trigger failure = no detector crossing by deadline. Stop begins only after
    detection, so a detection manipulation changes both latency and failures.
    """
    mg,md,ms=drifts;z,w=nodes(n)
    td=(z+1)*deadline/2;wd=w*deadline/2
    limit=np.maximum(horizon-ssd-td,0)
    ts=(z[None,:]+1)*limit[:,None]/2;ws=w[None,:]*limit[:,None]/2
    finish=ssd+td[:,None]+ts
    joint=(fpt_pdf(td,md,noises[1])*wd)[:,None]*fpt_pdf(ts,ms,noises[2])*ws
    fg=go_cdf(finish,mg,noises[0],go_boundary,initial_variance);fgH=float(go_cdf(horizon,mg,noises[0],go_boundary,initial_variance))
    prevented=float(np.sum(joint*(fgH-fg)))
    response=fgH-prevented
    cancel=float(np.sum(joint*(1-fg)))
    return dict(response_probability=response,cancel_probability=cancel,unresolved_probability=1-response-cancel,
        trigger_failure=1-float(fpt_cdf(deadline,md,noises[1])),
        detection_conditional_median_s=conditional_median(md,noises[1],1.,deadline),
        stop_mean_after_trigger_s=1/ms if ms>0 else None,
        go_RMST_s=go_restricted_mean(mg,noises[0],go_boundary,initial_variance,horizon))


def sample_fpt(rng,n,mu,sigma,boundary=1.):
    if mu==0:return (boundary/sigma/rng.normal(size=n))**2
    times=rng.wald(boundary/abs(mu),boundary**2/sigma**2,size=n)
    if mu<0:times=np.where(rng.random(n)<np.exp(2*mu*boundary/sigma**2),times,np.inf)
    return times


def monte_carlo_check(drifts,ssd,seed,n=200000,go_boundary=1.,noises=tuple(NOISE),initial_variance=0.):
    rng=np.random.default_rng(seed)
    boundary=go_boundary-rng.normal(size=n)*np.sqrt(initial_variance)
    go=sample_fpt(rng,n,drifts[0],noises[0],np.maximum(boundary,1e-12));go[boundary<=0]=0.
    det,stop=[sample_fpt(rng,n,mu,sigma) for mu,sigma in zip(drifts[1:],noises[1:])]
    trigger=det<=DETECTION_DEADLINE
    response,cancel,unresolved=stop_race(go,det+stop,trigger,ssd,HORIZON)
    return dict(seed=seed,n=n,response=float(response.mean()),cancel=float(cancel.mean()),unresolved=float(unresolved.mean()),
                trigger_failure=float((~trigger).mean()))


def run():
    source=pd.read_csv(OUT/'molecular.csv')
    # Both tensor choices and all three nuclear damping scenarios propagate.
    selected=source[(source.retention==1)&(source.cycles==20)]
    nr=[];ar=[]
    baseline=[race_probabilities(tuple(DRIFTS),ssd) for ssd in SSDS]
    for _,s in selected.iterrows():
        for gain in (-100.,0.,10.,100.):
            ns=spike_metrics(s.product_fraction,s.reference_product_fraction,gain)
            n0=spike_metrics(s.reference_product_fraction,s.reference_product_fraction,gain)
            common=dict(basis=s.basis,gamma_oxygen_s=s.gamma_oxygen_s,gamma_nuclear_s=s.gamma_nuclear_s,model=s.model,Gmax_pA=gain,
                product_fraction=s.product_fraction,reference_product_fraction=s.reference_product_fraction)
            nr.append(dict(**common,**ns,delta_firing_probability=ns['firing_probability']-n0['firing_probability'],
                delta_first_spike_RMST_s=ns['first_spike_RMST_s']-n0['first_spike_RMST_s'],
                delta_rate_Hz=ns['renewal_rate_Hz']-n0['renewal_rate_Hz']))
            for coupling in (.1,1.):
                for neff in (100.,1000.):
                    v0=n0['renewal_count_variance_per_s'];v=ns['renewal_count_variance_per_s']
                    noise0=np.sqrt(NOISE**2+coupling**2*v0/neff)
                    var0=coupling**2*v0*.2/neff
                    ref=[race_probabilities(tuple(DRIFTS),ssd,noises=tuple(noise0),initial_variance=var0) for ssd in SSDS]
                    for role in ROLES:
                        dr=ns['renewal_rate_Hz']-n0['renewal_rate_Hz']
                        mus=DRIFTS.copy();boundary=1.;noises=noise0.copy();var=var0
                        if role=='preparation':
                            boundary=1.-coupling*dr*.2;var=coupling**2*v*.2/neff
                        else:
                            j=('go','detection','stop').index(role)
                            mus[j]+=coupling*dr
                            noises[j]=np.sqrt(NOISE[j]**2+coupling**2*v/neff)
                        tail=mixture_error_bound(mus[0],noises[0],boundary,var)
                        if boundary<=0 or tail>1e-8:raise ValueError('premature go requires a full absorbing pre-go model')
                        for k,ssd in enumerate(SSDS):
                            r=race_probabilities(tuple(mus),ssd,go_boundary=boundary,noises=tuple(noises),initial_variance=var)
                            ar.append(dict(**common,role=role,neural_circuit_gain=coupling,effective_neurons=neff,ssd_s=ssd,
                                go_boundary=boundary,initial_variance=var,initial_mixture_error_bound=tail,
                                go_noise=noises[0],detection_noise=noises[1],stop_noise=noises[2],
                                go_drift=mus[0],detection_drift=mus[1],stop_drift=mus[2],**r,
                                delta_response_probability=r['response_probability']-ref[k]['response_probability'],
                                delta_trigger_failure=r['trigger_failure']-ref[k]['trigger_failure'],
                                delta_detection_conditional_median_s=r['detection_conditional_median_s']-ref[k]['detection_conditional_median_s'],
                                delta_stop_mean_s=r['stop_mean_after_trigger_s']-ref[k]['stop_mean_after_trigger_s'],
                                delta_go_RMST_s=r['go_RMST_s']-ref[k]['go_RMST_s']))
    pd.DataFrame(nr).to_csv(OUT/'neuron.csv',index=False)
    pd.DataFrame(ar).to_csv(OUT/'action.csv',index=False)
    save(OUT/'baseline.json',dict(drifts=DRIFTS.tolist(),noise=NOISE.tolist(),SSDs=SSDS,outcomes=baseline,
         neuron=spike_metrics(.01,.01,0.),neuronal_parameters=dict(C_pF=20,I0_pA=4,margin_mV=10,noise_mV_sqrt_s=20,half_product_fraction=.02,refractory_s=.005),
         circuit_gain_units='accumulator units per spike; population/cell-to-circuit mapping unmeasured',
         preparation='initial mean go offset = coupling * delta_rate * .2 s; variance = coupling^2 * count_variance_rate * .2 / Neff; above-boundary tail bounded explicitly',
         population_noise='renewal diffusion approximation: variance_rate=a*sigmaV^2/(a+mu*tref)^3; circuit noise sqrt(intrinsic^2+kappa^2*variance_rate/Neff)',
         effective_neurons=[100,1000],effective_neuron_status='design scenarios, not observations; Neff=N/[1+(N-1)rho] only under equicorrelated approximation',
         independence='separate neuronal populations feed go/detection/stop and preparation; common-input circuit correlations not calibrated',
         physical_scope='zero-leak perfect integrate-and-fire neuron and population diffusion circuit under held messenger plateau; no human EEG/SSRT fit',
         source_to_neuron='P-formation writes an unmeasured downstream messenger M; initial M=.05*yield. Gmax pA current capacity; not literal persistent oxidized-FAD occupancy.',
         readout_timing='M held from .2s before go through the 3s circuit window as a theoretical plateau. A long-lived messenger or maintained flux is required; cofactor P recovery alone does not provide it. Finite-lifetime bounds saved separately.'))
    precision=[]
    zcrit=norm.ppf(1-.05/(2*20))+norm.ppf(.8)
    for d in (1e-4,1e-3,.01,.05):
        for paired_sd in (.1,.5,1.):
            for icc in (0.,.1):
                n=(zcrit*paired_sd/d)**2*(1+19*icc)
                precision.append(dict(target_probability_difference=d,SD_of_paired_binary_difference=paired_sd,ICC=icc,trials_per_animal=20,
                     alpha=.05,power=.8,comparisons=20,required_trial_pairs=n,required_animals=n/20,
                     status='normal-approximation precision curve; SD/ICC unmeasured, 20 comparisons only for a preselected source/gain/model contrast'))
    pd.DataFrame(precision).to_csv(OUT/'precision.csv',index=False)
    # Required current capacity for a +5 percentage-point spike probability.
    req=[]
    for margin in (5.,10.,20.):
        for noise in (5.,20.,50.):
            base=float(fpt_cdf(.05,200.,noise,margin));target=base+.05
            if target>=1:continue
            mu=brentq(lambda mu:float(fpt_cdf(.05,mu,noise,margin))-target,200.,10000.)
            for delta in (1e-6,1e-5,1e-4,1e-3,.01):
                pref=.01;dg=(pref+delta)/(.02+pref+delta)-pref/(.02+pref)
                di=(mu-200.)*20/1000
                req.append(dict(margin_mV=margin,noise_mV_sqrt_s=noise,delta_product_fraction=delta,
                    baseline_probability=base,target_probability=target,required_delta_current_pA=di,required_Gmax_pA=di/dg))
    pd.DataFrame(req).to_csv(OUT/'required_gain.csv',index=False)
    print('neuron rows',len(nr),'action rows',len(ar),'required gains',len(req),flush=True)


if __name__=='__main__':run()
