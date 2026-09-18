from pathlib import Path
import sys,json,hashlib,time
from dataclasses import asdict,replace
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl import stochastic_branch_spike as s
from impl import branch_current_gain as b
from calibrate import calibrate
HERE=Path(__file__).resolve().parent;OUT=ROOT/'data/stochastic-branch-spike';OUT.mkdir(exist_ok=True)
def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def table(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False)
old=json.loads((ROOT/'data/branch-current-gain/preserved_inputs.json').read_text())['records']
old+=json.loads((ROOT/'audit/branch-current-gain-manifest.json').read_text());old=list({x['path']:x for x in old}.values())
for x in old:assert hashlib.sha256((ROOT/x['path']).read_bytes()).hexdigest()==x['sha256'],x['path']
save('preserved_inputs.json',dict(records=old))
source=json.loads((ROOT/'data/branch-current-gain/summary.json').read_text());y0=source['reference_yield'];dy=source['delta_yield']
seeds=[2026091711,2026091712,2026091713];true=s.Noise();p0=b.Scenario(tau_gate_s=.005)
save('configuration.json',dict(seeds=seeds,source='data/branch-current-gain/summary.json',reference_yield=y0,history_delta_yield=dy,
    baseline_noise=asdict(true),baseline_membrane=asdict(p0),dt_s=.0001,trials_per_arm_seed=8192,sensitivity_trials=4096,
    preparation='60 ms common voltage-clamped equilibration; at t=0 replace finite P/E pool with selected branch mixture; hold h/channels continuous; release voltage with current step',
    chemistry='fixed N pool, exact two-state refresh toward y0, equal P/E lifetime only; stochastic realization of prior mean law, not identified chemical mechanism',
    channel='N two-state channels, conditional q set by new response law, relaxation tau; frozen-q exact count transition',
    threshold='first passage of a passive membrane with finite conductance; no reset or action-potential waveform; right censor at 50ms',
    coefficient_status='all physical values assumed or recovered from synthetic calibration only',
    calibration='16 independent synthetic blocks fit; 8 held-out; history firing absent from all calibration fits',
    uncertainty='paired common random numbers quantify Monte Carlo error only; calibration seed variability kept separate; no native prior distribution',
    physical_measurements=0,native_noise_parameters=None,native_firing_distribution=None))
distributions=[];contrasts=[];fits=[];cdfrows=[];paths={};started=time.monotonic()
def record(tag,response,index,p,noise,n,molecular,channel,dt=.0001,seed_offset=0):
    seed=seeds[index]+seed_offset
    tr=s.simulate_pair(y0,dy,p,n=n,seed=seed,noise=noise,dt=dt,molecular=molecular,channel=channel)
    name=f'{response}__{tag}__seed{seeds[index]}'
    path=OUT/(name+'.npz');np.savez_compressed(path,**tr)
    paths[(response,tag,index)]=path
    common=dict(case=tag,response=response,seed=seeds[index],dt_s=dt,molecules=noise.molecules,channels=noise.channels,
        molecular_noise=molecular,channel_noise=channel)
    for arm in [0,1]:
        distributions.append(dict(**common,arm='reference' if arm==0 else 'history',**s.distribution(tr['first_s'][arm])))
        for t in np.linspace(0,.05,101):cdfrows.append(dict(case=tag,response=response,seed=seeds[index],arm=arm,time_s=t,cdf=float(np.mean(tr['first_s'][arm]<=t))))
    contrasts.append(dict(**common,**s.contrast(tr['first_s'])))
    print(f'{name}: dP={contrasts[-1]["delta_probability"]:.6g}; MCSE={contrasts[-1]["paired_MC_SE"]:.3g}; elapsed={time.monotonic()-started:.1f}s',flush=True)
    # Periodic durable progress, without exposing partial outputs as calibrated results.
    table('distribution_by_seed.csv',distributions);table('contrast_by_seed.csv',contrasts)
    return tr

for index,seed in enumerate(seeds):
    params,rows=calibrate(seed+10000,y0,OUT/f'calibration_seed{seed}');fits.extend(rows);table('calibration_fits.csv',fits)
    for response,cap in [('linear_gate',.2),('bounded_amplifier',3.125)]:
        p=replace(p0,channel_response=response,capacity_nS=cap)
        for tag,molecular,channel in [('both',True,True),('molecule_only',True,False),('channel_only',False,True),('background_only',False,False)]:
            record(tag,response,index,p,true,8192,molecular,channel,seed_offset=20000)
        fp=b.Scenario(**params[response]['scenario']);fn=s.Noise(**params[response]['noise'])
        record('independent_fit_prediction',response,index,fp,fn,8192,True,True,seed_offset=30000)
        for tag,nn in [('molecules_1000',replace(true,molecules=1000)),('molecules_100000',replace(true,molecules=100000)),
                       ('channels_20',replace(true,channels=20)),('channels_2000',replace(true,channels=2000))]:
            record(tag,response,index,p,nn,4096,True,True,seed_offset=40000)
        for dt in [.0002,.00005]:
            record(f'dt_{dt:g}',response,index,p,true,4096,True,True,dt=dt,seed_offset=50000)

table('cdf_by_seed.csv',cdfrows)
validation=[]
for index in range(3):
    for response in ['linear_gate','bounded_amplifier']:
        raw=np.load(paths[(response,'both',index)])['first_s'];pred=np.load(paths[(response,'independent_fit_prediction',index)])['first_s']
        for arm in range(2):
            grid=np.linspace(0,.05,1001)
            a=np.mean(raw[arm,:,None]<=grid,axis=0);c=np.mean(pred[arm,:,None]<=grid,axis=0)
            # Union over both empirical CDFs in all 12 comparisons.
            dkw=2*np.sqrt(np.log(4*12/.05)/(2*raw.shape[1]))
            validation.append(dict(response=response,seed=seeds[index],arm=arm,
                heldout_CDF_max_error=float(np.max(abs(a-c))),family_DKW_two_sample_bound=float(dkw),
                within_DKW=bool(np.max(abs(a-c))<=dkw),probability_difference=float(c[-1]-a[-1]),
                RMST_difference_ms=float(np.mean(np.minimum(pred[arm],.05)-np.minimum(raw[arm],.05))*1000)))
table('heldout_prediction_checks.csv',validation)

# Pooled estimates preserve independent trial marginals; paired SEM is numerical.
pooled=[]
for response in ['linear_gate','bounded_amplifier']:
    for tag in ['both','molecule_only','channel_only','background_only','independent_fit_prediction','molecules_1000','molecules_100000','channels_20','channels_2000','dt_0.0002','dt_5e-05']:
        trials=np.concatenate([np.load(paths[(response,tag,i)])['first_s'] for i in range(3)],axis=1)
        d0=s.distribution(trials[0]);d1=s.distribution(trials[1]);delta=s.contrast(trials)
        seed_d=[x['delta_probability'] for x in contrasts if x['response']==response and x['case']==tag]
        pooled.append(dict(response=response,case=tag,reference_probability=d0['probability'],history_probability=d1['probability'],
            reference_RMST_ms=d0['RMST_ms'],history_RMST_ms=d1['RMST_ms'],reference_conditional_median_ms=d0['conditional_median_ms'],
            history_conditional_median_ms=d1['conditional_median_ms'],n_per_arm=trials.shape[1],
            seed_delta_mean=float(np.mean(seed_d)),seed_delta_std=float(np.std(seed_d,ddof=1)),**delta))
table('pooled_summary.csv',pooled)

save('identifiability.json',dict(separate_assays_required=['branch-resolved effective fraction and its correlation','fixed-ligand channel current and its correlation',
    'independent membrane current-step/noise response','dose/current curve and step kinetics for each relay'],
    equal_time_constant_counterexample='if molecular and channel current fluctuations share tau, covariance=(variance_m+variance_c)*exp(-lag/tau); firing/current covariance alone cannot determine the split',
    absolute_source_scale='A remains conditional; fraction noise identifies effective independent pool size only under the assumed observation and refresh model',
    calibration_idealizations='direct noise-free fraction observations and separated clamp controls; acquisition filters/detection noise absent',
    native_calibration=False))
save('execution_status.json',dict(status='simulation_complete',conditions=len(contrasts),distribution_rows=len(distributions),
    trials_per_arm_all_conditions=int(sum(x['n'] for x in distributions if x['arm']=='reference')),
    seeds=seeds,physical_measurements=0,native_parameters=None,preserved_files=len(old),elapsed_s=time.monotonic()-started))
for x in old:assert hashlib.sha256((ROOT/x['path']).read_bytes()).hexdigest()==x['sha256'],x['path']
print('Completed stochastic calibration and held-out firing predictions.',flush=True)
