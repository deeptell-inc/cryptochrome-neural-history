"""Preserve failed constant-step fit; improve independent membrane excitation."""
from pathlib import Path
import sys,json
from dataclasses import replace,asdict
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl import stochastic_branch_spike as s
from impl import branch_current_gain as b
OUT=ROOT/'data/stochastic-branch-spike'
config=json.loads((OUT/'configuration.json').read_text());y0=config['reference_yield'];dy=config['history_delta_yield']
fits=[];checks=[];pooled={};rows=[]
for seed in config['seeds']:
    rng=np.random.default_rng(seed+60000);dt=.0002;steps=10000;blocks=256
    # Independent random sign current every 5ms; membrane assay isolates g/noise.
    current=4*rng.choice([-1.,1.],size=(blocks,steps//25))
    v=np.zeros((blocks,steps+1));a=np.exp(-dt/.02);sd=20*np.sqrt((1-a*a)/100)
    for j in range(steps):v[:,j+1]=current[:,j//25]+(v[:,j]-current[:,j//25])*a+sd*rng.normal(size=blocks)
    # Sufficient statistics, complete trajectories retained; independent blocks split.
    train=192;X=np.column_stack([v[:train,:-1].ravel(),np.repeat(current[:train],25,axis=1).ravel()])
    af,beta=np.linalg.lstsq(X,v[:train,1:].ravel(),rcond=None)[0]
    rate=-np.log(af)/dt;gt=(1-af)/beta;C=1000*gt/rate
    residual=v[:train,1:].ravel()-X@np.array([af,beta]);sig=np.sqrt(np.mean(residual**2)*2*rate/(1-af*af))
    val=v[train:,1:]-af*v[train:,:-1]-beta*np.repeat(current[train:],25,axis=1)
    folder=OUT/f'calibration_seed{seed}'
    np.savez_compressed(folder/'membrane_upgrade_raw.npz',voltage_mV=v,current_blocks_pA=current,dt_s=dt,current_block_steps=25,fit_blocks=train)
    fits.append(dict(seed=seed,C_pF=C,total_g_nS=gt,sigma_mV_sqrt_s=sig,AR1=af,input_coefficient=beta,
        design_condition=float(np.linalg.cond(X)),validation_innovation_SD_mV=float(val.std()),
        validation_lag1_correlation=float(np.mean(val[:,:-1]*val[:,1:])/np.mean(val**2))))
    old=json.loads((folder/'fitted_parameters.json').read_text());new={}
    for response in ['linear_gate','bounded_amplifier']:
        p=replace(b.Scenario(**old[response]['scenario']),C_pF=float(C),gL_nS=float(gt-.2))
        noise=replace(s.Noise(**old[response]['noise']),background_sigma_mV_sqrt_s=float(sig))
        new[response]=dict(scenario=asdict(p),noise=asdict(noise))
        tr=s.simulate_pair(y0,dy,p,n=8192,seed=seed+70000,noise=noise)
        np.savez_compressed(OUT/f'{response}__upgraded_prediction__seed{seed}.npz',**tr)
        pooled.setdefault(response,[]).append(tr['first_s'])
        source=np.load(OUT/f'{response}__both__seed{seed}.npz')['first_s']
        for arm in range(2):
            grid=np.linspace(0,.05,1001)
            a0=np.mean(source[arm,:,None]<=grid,axis=0);a1=np.mean(tr['first_s'][arm,:,None]<=grid,axis=0)
            bound=2*np.sqrt(np.log(4*12/.05)/(2*source.shape[1]))
            gap=float(np.max(abs(a0-a1)))
            checks.append(dict(seed=seed,response=response,arm=arm,CDF_max_error=gap,DKW_bound=bound,
                within_DKW=gap<=bound,probability_difference=float(a1[-1]-a0[-1])))
        rows.append(dict(seed=seed,response=response,**s.contrast(tr['first_s'])))
        print(seed,response,C,gt,sig,rows[-1]['delta_probability'],flush=True)
    (folder/'upgraded_parameters.json').write_text(json.dumps(new,indent=2)+'\n')
pd.DataFrame(fits).to_csv(OUT/'upgraded_membrane_calibration.csv',index=False)
pd.DataFrame(checks).to_csv(OUT/'upgraded_prediction_checks.csv',index=False)
pd.DataFrame(rows).to_csv(OUT/'upgraded_contrast_by_seed.csv',index=False)
summary=[]
for response,parts in pooled.items():
    first=np.concatenate(parts,axis=1);d0=s.distribution(first[0]);d1=s.distribution(first[1])
    summary.append(dict(response=response,reference_probability=d0['probability'],history_probability=d1['probability'],
        reference_RMST_ms=d0['RMST_ms'],history_RMST_ms=d1['RMST_ms'],n_per_arm=first.shape[1],**s.contrast(first)))
pd.DataFrame(summary).to_csv(OUT/'upgraded_prediction_summary.csv',index=False)
# Observation-noise sensitivity of chemical-count calibration, separate from intrinsic noise.
obs=[]
for seed in config['seeds']:
    raw=np.load(OUT/f'calibration_seed{seed}/calibration_raw.npz')['molecular_fraction'][:16]
    rng=np.random.default_rng(seed+80000)
    for sd in [0.,.001,.003]:
        noisy=raw+rng.normal(0,sd,raw.shape);naive=s.fit_count_noise(noisy,y0,.0005,12)
        centered=noisy-y0;var=np.mean(centered**2)-sd**2;cov=np.mean(centered[:,:-12]*centered[:,12:]);rho=cov/var
        obs.append(dict(seed=seed,observation_SD_fraction=sd,naive_count=naive['effective_count'],naive_tau_s=naive['tau_s'],
             known_noise_corrected_count=y0*(1-y0)/var,known_noise_corrected_tau_s=-.006/np.log(rho)))
pd.DataFrame(obs).to_csv(OUT/'chemical_observation_noise_sensitivity.csv',index=False)
print('Independent membrane upgrade and observation-noise sensitivity complete.',flush=True)
