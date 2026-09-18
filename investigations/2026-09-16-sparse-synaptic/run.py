"""Prespecified synthetic synapse allocation and causal controls."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from scipy.stats import norm
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl import sparse_synaptic as s
from impl.dark_embedded_tensors import save

def main():
    s.OUT.mkdir(parents=True,exist_ok=True);sources=s.source_values();save(s.OUT/'sources.json',sources)
    records=[];races=[];proxies=[];registry=[]
    def run(label,source='full',gain=512,n=4096,seed=26091691,routes=('go','trigger','stop'),**kw):
        outputs={}
        for i,route in enumerate(routes):
            z=s.simulate(route,sources[source]['delta_yield'],gain,n,seed+{'go':0,'trigger':1000,'stop':2000}[route],**kw)
            outputs[route]=z
            for row in z['records']:row.update(run=label,source=source);records.append(row)
            if route=='go':proxies.append(dict(run=label,seed=seed,source=source,gain=gain,records=z['proxies']))
            registry.append(dict(run=label,route=route,source=source,seed=seed,**kw,weights=z['weights'].tolist(),targets=z['targets'].tolist(),
              scores=z['scores'].tolist(),allocation=z['allocation'].tolist(),subsets=z['subsets'],
              top_absolute_influence_fraction=float(abs(z['scores'][z['subsets']['top']]).sum()/abs(z['scores']).sum())))
            np.savez_compressed(s.OUT/f'{label}_{seed}_{gain}_{route}_trials.npz',rt=z['rt'],hits=z['hits'],mean_state=z['aggregate'],analytic_delta=z['analytic'])
        if len(outputs)==3:
            for row in s.race_records(**outputs):row.update(run=label,source=source,seed=seed,gain_probability_per_yield=gain,n=n);races.append(row)
        print(label,seed,gain,'done',flush=True)
        return outputs
    for seed in (26091691,26091692,26091693):
        for gain in (128,512):run('primary',gain=gain,n=8192,seed=seed)
    run('fast_oxygen',source='fast_oxygen',n=16384)
    run('reset',source='reset',n=1024)
    for arch in (26091683,26091685):run('architecture_'+str(arch),architecture=arch,n=4096)
    run('homogeneous',homogeneous=True,n=8192)
    run('short_memory',prep_leak=5.,routes=('go',),n=8192)
    run('route_block',coupling=0.,routes=('go',),n=8192)
    run('long_signal',tau=.05,n=4096)
    run('correlated',release_correlation=.2,n=8192)
    run('fine',dt=.001,n=8192)
    df=pd.DataFrame(records);df.to_csv(s.OUT/'neural_summary.csv',index=False)
    pd.DataFrame(races).to_csv(s.OUT/'race_summary.csv',index=False)
    save(s.OUT/'readiness_proxies.json',proxies);save(s.OUT/'circuit_registry.json',registry)
    group=['run','source','route','arm','gain_probability_per_yield','architecture','dt_s','tau_s','prep_leak_s','prep_to_go_coupling_s','homogeneous','release_correlation']
    numeric=['delta_RMST_s','mean_RMST_s','omission','delta_state_cue_aligned','analytic_delta_state_cue_aligned','expected_extra_release_count','observed_extra_release_count','proxy_delta_own_common','proxy_delta_fixed_common','proxy_alignment_component']
    df.groupby(group,dropna=False)[numeric].mean().reset_index().to_csv(s.OUT/'neural_aggregate.csv',index=False)
    pd.DataFrame(races).groupby(['run','source','route','arm','gain_probability_per_yield','SSD_s']).agg(delta_response=('delta_response','mean'),response_probability=('response_probability','mean'),trigger_failure=('trigger_failure','mean')).reset_index().to_csv(s.OUT/'race_aggregate.csv',index=False)
    precision=[]
    for effect in (.001,.005,.010):
      for sd in (.005,.02,.1):
       for rho in (0.,.1,.5):
        comparisons=45;m=20;z=norm.ppf(1-.05/(2*comparisons))+norm.ppf(.8);deff=1+(m-1)*rho
        precision.append(dict(target_effect_s=effect,assumed_paired_SD_s=sd,assumed_ICC=rho,trials_per_animal=m,comparisons=comparisons,alpha=.05,power=.8,required_trials=z*z*sd*sd/effect**2*deff,required_animals=z*z*sd*sd/effect**2*deff/m))
    pd.DataFrame(precision).to_csv(s.OUT/'measurement_precision.csv',index=False)
    save(s.OUT/'configuration.json',dict(species='synthetic circuit; molecular source is conditional dCRY-derived SQ model',
      synapses=64,active_sparse=8,presynaptic_opportunity_period_s=.01,baseline_release_probability=.25,
      weights='fixed seeded signed lognormal, absolute sum .45; fictional dimensionless jump amplitudes',
      source='delta P yield only; no density-matrix components read out',
      allocation='p_j(t)=p0+gain*delta_yield*alpha_j*exp(-(t-onset)/tau); sum alpha=1 for5 primary allocations',
      clocks='go source .203s after go task onset; trigger/stop .003s after their own activation; these are distinct event-locked hypothetical interventions',
      stop_clock_limit='stop arm assumes modulation delivered after trigger; not evidence that a cue-born12ms pulse survives detection latency',
      baseline_compensation='subtract p0 per opportunity to hold background mean fixed; same compensation in matched arms',
      noise='independent Gaussian OU plus Bernoulli release; shared random numbers across arms; optional .2 exchangeable release correlation',
      weights_selection='classical impulse response before molecular perturbation, independent of stochastic test outcomes',
      fair_budget='equal sum delta-p and opportunities, not equal postsynaptic weighted current, variance, or metabolic energy',
      block='remove modulation on8 edges without removing baseline transmission; do not renormalize lost budget; rescue restores it',
      EEG_mapping=None,native_CRY_to_synapse_mapping=None,native_gain=None,biological_replicates=0,
      precision='scenario curves only; MC SE and random seeds are not biological uncertainty',
      observation='input-aligned preparatory state; negativeP movement-aligned proxy with common-cohort control; no EEG voltage or SSRT calibration'))
if __name__=='__main__':main()
