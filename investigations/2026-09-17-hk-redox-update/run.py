"""Partial calibration using published cell-level summaries, never raw-trial reconstruction."""
from pathlib import Path
import csv
from dataclasses import asdict
import hashlib
import importlib.util
import json
import sys
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from impl.hk_redox_memory import EffectiveMemory, current_change_bounds

OUT = ROOT/'data/hk-redox-update'
OUT.mkdir(exist_ok=True)
SOURCE = ROOT/'evidence/2026-09-17-native-noise-literature'
spec = importlib.util.spec_from_file_location('source_reader', SOURCE/'extract_source_data.py')
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


def save(name, data):
    (OUT/name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def table(name, rows):
    with (OUT/name).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def ingest():
    sheets = reader.extract()
    settings = {
        'Fig. 4f': ('WT', False, ['C','D','E','F'], ['H','I','J','K'], [0,10,20,30]),
        'Fig. 6d': ('WT', True, ['C','D','F','G'], ['I','J','L','M'], [0,10,30,40]),
        'ED Fig. 5b': ('Hk_K289M_rescue', True, ['C','D','F'], ['H','I','K'], [0,10,30]),
        'ED Fig. 5d': ('Hk_WT_rescue', True, ['C','D','F'], ['H','I','K'], [0,10,30]),
    }
    rows=[]
    for sheet, (genotype, train, fast, slow, times) in settings.items():
        cells=sheets[sheet]
        for key, label in cells.items():
            if not key.startswith('A') or not label.startswith('Cell '):
                continue
            row=int(key[1:]); number=int(label.split()[-1]); cid=f'{sheet}:{label}'
            for mode, columns in [('fast',fast), ('slow',slow)]:
                assert cells[columns[0]+'1']==f't {mode} (ms)'
                baseline=cells.get(columns[0]+str(row))
                for col,t in zip(columns,times):
                    assert float(cells[col+'3'])==t
                    value=cells.get(col+str(row))
                    if value is not None:assert float(value)>0
                    split = 'control_only' if genotype!='WT' else ('fit' if number%2 else 'cell_validation')
                    if genotype=='WT' and t==40:
                        split='condition_validation' if number%2==0 else 'time_extrapolation_fit_cells'
                    rows.append(dict(species='Drosophila melanogaster',cell_system='adult dFBN',
                        genotype=genotype,source_doi='10.1038/s41586-025-08734-4',sheet=sheet,source_cell=col+str(row),
                        cell_id=cid,cell_number=number,animal_id=None,mode=mode,time_min=t,unit='ms',
                        tau_ms=None if value is None else float(value),baseline_tau_ms=None if baseline is None else float(baseline),
                        log_ratio=None if value is None or baseline is None else float(np.log(float(value)/float(baseline))),
                        split=split,train_10_to_30_min=train,nominal_pipette_4ONE_uM=50,
                        local_4ONE_uM=None,measurement_level='published_per_cell_fitted_time_constant',
                        independent_trial_variance=None,observation_fit_error=None))
    table('observations.csv', rows);save('observations.json',rows)
    return rows,sheets


def unpack(theta, model):
    if model=='dynamic':return EffectiveMemory(np.exp(theta[0]),np.exp(theta[1]),theta[2],theta[3],theta[4])
    if model=='no_voltage':return EffectiveMemory(np.exp(theta[0]),0.,0.,theta[1],theta[2])
    return None


def predict(theta, model, rows):
    if model=='static':
        return np.array([theta[0 if r['mode']=='fast' else 1]
                         if r['time_min']>0 and r['genotype']!='Hk_K289M_rescue' else 0 for r in rows])
    p=unpack(theta,model)
    # Vectorized protocol/mode indexing also preserves missing-data row order.
    t=np.array([r['time_min'] for r in rows]);train=np.array([r['train_10_to_30_min'] for r in rows])
    x=p.coordinate(t,False);x[train]=p.coordinate(t[train],True)
    amplitudes=np.array([p.log_tau_fast_excursion if r['mode']=='fast' else p.log_tau_slow_excursion for r in rows])
    amplitudes[np.array([r['genotype']=='Hk_K289M_rescue' for r in rows])]=0
    return amplitudes*x


BOUNDS={
    'dynamic':(np.array([np.log(1e-4),np.log(1e-4),0.,0.,0.]),np.array([np.log(10),np.log(100),10.,4.,4.])),
    'no_voltage':(np.array([np.log(1e-4),0.,0.]),np.array([np.log(10),4.,4.])),
    'static':(np.zeros(2),np.full(2,4.)),
}


def weights(rows, scales):
    counts={}
    for r in rows:
        key=(r.get('draw_id',r['cell_id']),r['mode']);counts[key]=counts.get(key,0)+1
    return np.array([1/(scales[r['mode']]*np.sqrt(counts[(r.get('draw_id',r['cell_id']),r['mode'])])) for r in rows])


def fit(rows,model,scales,start=None,bounds=None):
    y=np.array([r['log_ratio'] for r in rows]);w=weights(rows,scales)
    lo,hi=BOUNDS[model] if bounds is None else bounds
    if start is not None:starts=[start]
    elif model=='dynamic':starts=[np.array([np.log(k),np.log(v),B,.5,.8]) for k,v,B in [(.01,.05,.5),(.1,.1,.5),(1,1,.5),(.01,1,2),(.5,.01,.1)]]
    elif model=='no_voltage':starts=[np.array([np.log(k),.5,.8]) for k in [.01,.1,1]]
    else:starts=[np.array([.3,.5])]
    best=None
    for initial in starts:
        result=least_squares(lambda theta:(predict(theta,model,rows)-y)*w,np.clip(initial,lo+1e-9,hi-1e-9),
                             bounds=(lo,hi),max_nfev=1500,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        if best is None or result.cost<best.cost:best=result
    return best


def metrics(rows,prediction,scales):
    y=np.array([r['log_ratio'] for r in rows]);errors=prediction-y
    cells=sorted({r['cell_id'] for r in rows});bycell=[]
    for c in cells:
        terms=[]
        for mode in ['fast','slow']:
            mask=np.array([r['cell_id']==c and r['mode']==mode for r in rows])
            if mask.any():terms.append(float(np.mean((errors[mask]/scales[mode])**2)))
        bycell.append(np.mean(terms))
    return dict(cell_balanced_standardized_RMSE=float(np.sqrt(np.mean(bycell))),
                unweighted_log_RMSE=float(np.sqrt(np.mean(errors**2))),cells=len(cells),points=len(rows))


def main():
    rows,sheets=ingest()
    observed=[r for r in rows if r['log_ratio'] is not None and r['time_min']>0]
    train=[r for r in observed if r['split']=='fit']
    scales={m:float(np.std([r['log_ratio'] for r in train if r['mode']==m],ddof=1)) for m in ['fast','slow']}
    fits={};predictions=[];comparison=[]
    for model in BOUNDS:
        f=fit(train,model,scales);theta=f.x;fits[model]=dict(theta=theta.tolist(),cost=float(f.cost),
            success=bool(f.success),status=f.message,parameters=None if model=='static' else asdict(unpack(theta,model)))
        for split in ['fit','cell_validation','condition_validation','time_extrapolation_fit_cells','control_only']:
            subset=[r for r in observed if r['split']==split]
            pred=predict(theta,model,subset)
            comparison.append(dict(model=model,split=split,**metrics(subset,pred,scales)))
            for r,y in zip(subset,pred):predictions.append(dict(model=model,cell_id=r['cell_id'],sheet=r['sheet'],
                genotype=r['genotype'],mode=r['mode'],time_min=r['time_min'],split=split,
                observed_log_ratio=r['log_ratio'],predicted_log_ratio=float(y),baseline_tau_ms=r['baseline_tau_ms'],
                predicted_tau_ms=float(r['baseline_tau_ms']*np.exp(y)),observed_tau_ms=r['tau_ms']))
    save('fits.json',dict(fits=fits,scales=scales,search_bounds={k:[lo.tolist(),hi.tolist()] for k,(lo,hi) in BOUNDS.items()},
        scope='conditional effective dynamics; log observation law; bounds are numerical search ranges, not physiological priors'))
    table('prediction_rows.csv',predictions);table('model_comparison.csv',comparison)
    save('model_comparison.json',comparison)
    theta=np.array(fits['dynamic']['theta']);base=unpack(theta,'dynamic')
    print('fits',json.dumps(fits),flush=True)
    # Whole-cell resampling within each independent figure cohort. Each sampled
    # cell carries all its modes/times and has a new bootstrap cluster label.
    rng=np.random.default_rng(2026091741);boot=[];expanded_boot=[]
    lo,hi=BOUNDS['dynamic'];lo2=lo.copy();hi2=hi.copy();lo2[:2]-=np.log(10);hi2[:2]+=np.log(10);hi2[2:]*=2
    groups={s:sorted({r['cell_id'] for r in train if r['sheet']==s}) for s in ['Fig. 4f','Fig. 6d']}
    for rep in range(200):
        sample=[]
        for sheet,cids in groups.items():
            for draw,cid in enumerate(rng.choice(cids,len(cids),replace=True)):
                sample.extend(dict(r,draw_id=f'{sheet}:{draw}') for r in train if r['cell_id']==cid)
        f=fit(sample,'dynamic',scales,start=theta)
        # Keep failures and bound hits; do not silently discard weak fits.
        boot.append(dict(rep=rep,success=bool(f.success),cost=float(f.cost),**asdict(unpack(f.x,'dynamic'))))
        wider=fit(sample,'dynamic',scales,start=f.x,bounds=(lo2,hi2))
        expanded_boot.append(dict(rep=rep,success=bool(wider.success),cost=float(wider.cost),**asdict(unpack(wider.x,'dynamic'))))
    table('cell_bootstrap.csv',boot)
    table('expanded_bound_bootstrap.csv',expanded_boot)
    intervals={key:np.percentile([r[key] for r in boot],[2.5,50,97.5]).tolist() for key in asdict(base)}
    save('bootstrap_summary.json',dict(repetitions=len(boot),seed=2026091741,successful=sum(r['success'] for r in boot),
        percentile_intervals=intervals,scope='cell-cluster resampling conditional on model, independent cells and observed baselines; no animal/batch IDs',
        near_upper_bounds={key:int(sum(abs(r[key]-bound)<.001*bound for r in boot)) for key,bound in
            [('k_rest_per_min',10),('extra_train_per_min',100),('baseline_over_excursion',10),
             ('log_tau_fast_excursion',4),('log_tau_slow_excursion',4)]},
        expanded_bound_percentile_intervals={key:np.percentile([r[key] for r in expanded_boot],[2.5,50,97.5]).tolist() for key in asdict(base)}))
    # Profile effective rest rate; these are loss curves, not likelihood CIs.
    profiles=[]
    y=np.array([r['log_ratio'] for r in train]);w=weights(train,scales)
    for rate in np.geomspace(1e-4,10,41):
        def residual(v):return (predict(np.r_[np.log(rate),v],'dynamic',train)-y)*w
        lo,hi=BOUNDS['dynamic'];f=least_squares(residual,theta[1:],bounds=(lo[1:],hi[1:]),max_nfev=2000)
        profiles.append(dict(k_rest_per_min=float(rate),weighted_SSE=float(2*f.cost),success=bool(f.success),
                             extra_train_per_min=float(np.exp(f.x[0])),B=float(f.x[1])))
    table('rate_profile.csv',profiles)
    expanded=fit(train,'dynamic',scales,start=theta,bounds=(lo2,hi2))
    save('boundary_sensitivity.json',dict(parameters=asdict(unpack(expanded.x,'dynamic')),cost=float(expanded.cost),
        max_prediction_difference_log_tau=float(np.max(abs(predict(theta,'dynamic',observed)-predict(expanded.x,'dynamic',observed)))),
        scope='expanded numerical search bounds, not biological uncertainty'))
    # Absolute redox occupancy and derivative gains are indistinguishable.
    family=[dict(excursion=D,**base.physical_realization(D)) for D in np.array([.02,.2,.8])/(1+base.baseline_over_excursion)]
    save('occupation_family.json',family)
    # Reuse exactly the referenced ED Fig. 4c cohort, with an explicit duplicate check.
    c=sheets['ED Fig. 4c'];mainc=sheets['Fig. 4f'];currentrows=[]
    for row in range(4,15):
        erow=row+1
        for col in ['C','D','H','I']:
            assert float(c[col+str(erow)])==float(mainc[col+str(row)])
        tf0,tf1,ts0,ts1=[float(mainc[col+str(row)]) for col in ['C','D','H','I']]
        pk0,pk1=[float(c[col+str(erow)]) for col in ['M','N']]
        for t in [0.,5.,20.,50.,100.]:
            for scenario,pk in [('matched_peak',pk0),('observed_peak_including_rundown',pk1)]:
                for shared in (True,False):
                    lower,upper=current_change_bounds(t,pk0,pk,(tf0,ts0),(tf1,ts1),shared_weight=shared)
                    currentrows.append(dict(cell_id=f'Fig. 4f:Cell {row-3}',time_after_peak_ms=t,scenario=scenario,
                        mixture_assumption='unchanged' if shared else 'independently_changing',
                        peak_before_pA=pk0,peak_after_pA=pk,lower_delta_outward_pA=float(lower),upper_delta_outward_pA=float(upper),
                        fast_weight_min=0,fast_weight_max=1,kind='conditional_reconstruction_not_measured_current_difference'))
    table('current_envelopes.csv',currentrows)
    save('coefficient_status.json',dict(
        dFB_fit='effective 50 uM pipette exposure and whole voltage-train protocol only; conditional log-tau observation',
        measured='published per-cell tau, I_A peak, Rm, tau_m; independent raw variance unavailable',
        native_dark_CRY_branch_to_local_concentration_uM=None,native_Hk_redox_fraction=None,
        native_local_concentration_to_oxidation_rate=None,native_voltage_exchange_curve=None,
        native_channel_count=None,native_single_channel_current_pA=None,native_channel_noise=None,
        native_molecular_noise=None,native_lLNv_gain_pA_per_branch_fraction=None,native_firing_probability_difference=None,
        go_trigger_stop_gains=None,
        warning='Old synthetic ~0.14 or ~2.4 pp differences are preserved scenarios and are not recalibrated biological estimates.'))
    save('data_provenance.json',dict(source_doi='10.1038/s41586-025-08734-4',
        source_file=str((SOURCE/'41586_2025_8734_MOESM4_ESM.xlsx').relative_to(ROOT)),
        sha256=hashlib.sha256((SOURCE/'41586_2025_8734_MOESM4_ESM.xlsx').read_bytes()).hexdigest(),
        unit='cell-level fitted channel time constant (ms)',absolute_chemical_occupancy_measured=False,
        source_license='CC BY 4.0 per article license; original attribution retained',
        exclusions='missing tau or baseline excluded from ratio fit; no imputation; t0 conditions predictions only',
        source_rows=len(rows),missing_values=sum(r['tau_ms'] is None for r in rows),
        main_train_cells=len({r['cell_id'] for r in train}),physical_experiments_performed=0,
        time_window='chemical min; post-peak electrical ms; no nuclear relaxation inference'))
    print('Partial dFB fit, cell holdout, bootstrap and current envelopes complete.',flush=True)


if __name__=='__main__':main()
