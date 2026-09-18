from pathlib import Path
import csv
from dataclasses import asdict
import importlib.util
import json
import math
import os
import tempfile
import sys
import numpy as np
from scipy.integrate import solve_ivp

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl.hk_redox_memory import EffectiveMemory, a_current, current_change_bounds
HERE=Path(__file__).resolve().parent;OUT=ROOT/'data/hk-redox-update'
spec=importlib.util.spec_from_file_location('hk_run',HERE/'run.py');run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)


def main():
    fits=json.loads((OUT/'fits.json').read_text());obs=json.loads((OUT/'observations.json').read_text())
    p=EffectiveMemory(**fits['fits']['dynamic']['parameters'])
    predictions=list(csv.DictReader((OUT/'prediction_rows.csv').open()))
    # Verify independently by stepping the physical 2-state ODE at each phase.
    maximum=0.;family=json.loads((OUT/'occupation_family.json').read_text())
    for member in family:
        for train in [False,True]:
            times=[0,10,20,30,40];z=member['z0'];zs=[z]
            for start,end in zip(times[:-1],times[1:]):
                b=member['exchange_rest_per_min']+(member['extra_train_per_min'] if train and 10<=start<30 else 0)
                a=member['oxidation_per_min']
                sol=solve_ivp(lambda t,x:a*(1-x)-b*x,(start,end),[z],method='DOP853',rtol=1e-11,atol=1e-13)
                assert sol.success
                z=sol.y[0,-1];zs.append(z)
            reference=(np.array(zs)-member['z0'])/member['excursion']
            maximum=max(maximum,float(np.max(abs(reference-p.coordinate(times,train)))))
            assert min(zs)>=0 and max(zs)<=1
    assert maximum<1e-9
    train_ids={r['cell_id'] for r in obs if r['split']=='fit'}
    test_ids={r['cell_id'] for r in obs if r['split'] in ('cell_validation','condition_validation')}
    assert train_ids.isdisjoint(test_ids)
    assert not any(r['time_min']==40 for r in obs if r['split']=='fit')
    assert all(r['source_cell'] and r['unit']=='ms' for r in obs)
    # Independent scalar log calculations; source reader uses numpy log.
    log_error=max(abs(math.log(r['tau_ms']/r['baseline_tau_ms'])-r['log_ratio'])
                  for r in obs if r['log_ratio'] is not None)
    # Descriptive model comparison uncertainty across entire held-out cells.
    rng=np.random.default_rng(2026091742);comparisons=[]
    scales=fits['scales']
    for split in ['cell_validation','condition_validation']:
        rows=[r for r in predictions if r['split']==split and r['model'] in ('dynamic','static')]
        cids=sorted({r['cell_id'] for r in rows});loss={}
        for model in ['dynamic','static']:
            for cid in cids:
                parts=[]
                for mode in ['fast','slow']:
                    rr=[r for r in rows if r['cell_id']==cid and r['model']==model and r['mode']==mode]
                    if rr:parts.append(sum(((float(r['predicted_log_ratio'])-float(r['observed_log_ratio']))/scales[mode])**2 for r in rr)/len(rr))
                loss[(model,cid)]=sum(parts)/len(parts)
        diff=np.array([loss[('dynamic',c)]-loss[('static',c)] for c in cids])
        boot=rng.choice(diff,(4000,len(diff)),replace=True).mean(1)
        comparisons.append(dict(split=split,cells=len(cids),mean_dynamic_minus_static_MSE=float(diff.mean()),
            cluster_percentile_interval=np.percentile(boot,[2.5,97.5]).tolist(),
            scope='descriptive conditional on independent cells, no animal IDs; positive favors static'))
    # Check recoverability inside the assumed family before using real-data fit.
    true=EffectiveMemory(.04,.12,.7,.5,1.2)
    synth=[]
    for sheet,train in [('rest',False),('train',True)]:
        for mode in ['fast','slow']:
            for t in [2,5,10,20,30,40]:
                synth.append(dict(cell_id=sheet,mode=mode,time_min=t,train_10_to_30_min=train,genotype='WT',
                    log_ratio=float(true.log_tau_ratio(t,mode,train))))
    recovered=run.fit(synth,'dynamic',{'fast':1.,'slow':1.})
    fit_error=max(abs(np.array(list(asdict(run.unpack(recovered.x,'dynamic')).values()))/np.array(list(asdict(true).values()))-1))
    assert fit_error<1e-6
    # Common-mixture current envelope: both endpoints suffice; verify by sampling.
    current=list(csv.DictReader((OUT/'current_envelopes.csv').open()));current_summary=[]
    for scenario in ['matched_peak','observed_peak_including_rundown']:
        for mixture in ['unchanged','independently_changing']:
            selected=[r for r in current if float(r['time_after_peak_ms'])==20 and r['scenario']==scenario and r['mixture_assumption']==mixture]
            current_summary.append(dict(scenario=scenario,time_ms=20,cells=len(selected),mixture_assumption=mixture,
                mean_lower_delta_outward_pA=sum(float(r['lower_delta_outward_pA']) for r in selected)/len(selected),
                mean_upper_delta_outward_pA=sum(float(r['upper_delta_outward_pA']) for r in selected)/len(selected),
                interpretation='conditional ensemble envelope with cell-specific weights; not observed DeltaI'))
    summary=dict(independent_physical_ODE_coordinate_error=maximum,independent_scalar_log_error=log_error,
        synthetic_parameter_recovery_max_relative_error=float(fit_error),train_test_cells_disjoint=True,
        reaccumulation_condition_not_fit=True,heldout_model_comparison=comparisons,current_summary=current_summary,
        end_to_end_native_gain_identified=False,default_replacement_of_old_spiking_model=False)
    run.save('verification.json',summary)
    # Figures show observable ratios, never infer an absolute redox fraction.
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir()) / 'hk-redox-mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    bootstrap=list(csv.DictReader((OUT/'cell_bootstrap.csv').open()))
    grid=np.linspace(0,40,201)
    for i,mode in enumerate(['fast','slow']):
        for j,(sheet,train) in enumerate([('Fig. 4f',False),('Fig. 6d',True)]):
            ax=axes[i,j]
            estimates=np.array([EffectiveMemory(**{k:float(b[k]) for k in asdict(p)}).log_tau_ratio(grid,mode,train) for b in bootstrap])
            lower,upper=np.percentile(estimates,[2.5,97.5],axis=0)
            ax.fill_between(grid,lower,upper,color='#2b6d9c',alpha=.15,label='Cell bootstrap of mean model')
            ax.plot(grid,p.log_tau_ratio(grid,mode,train),color='#2b6d9c',label='Dynamic candidate')
            static=fits['fits']['static']['theta'][i]
            ax.hlines(static,.1,40,color='#bd7548',linestyle='--',label='Static comparator')
            for split,marker,color in [('fit','o','#444444'),('cell_validation','s','#da9834'),
                                        ('condition_validation','^','#b84b59'),('time_extrapolation_fit_cells','x','#898989')]:
                rr=[r for r in obs if r['sheet']==sheet and r['mode']==mode and r['split']==split and r['log_ratio'] is not None and r['time_min']>0]
                if rr:ax.scatter([r['time_min']+(r['cell_number']%5-2)*.15 for r in rr],[r['log_ratio'] for r in rr],
                                 marker=marker,color=color,s=19,alpha=.6,label=split.replace('_',' '))
            if train:ax.axvspan(10,30,color='#e8cc77',alpha=.2)
            ax.axhline(0,color='grey',lw=.6)
            ax.set(title=f'{sheet}: {mode} inactivation',ylabel='log(tau / baseline tau)',xlabel='Chemical protocol time (min)')
    handles,labels=axes[0,1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8)
    fig.suptitle('dFB / 50 uM pipette 4-ONE: partial calibration; cell-level summaries')
    fig.tight_layout(rect=(0,.12,1,.94));fig.savefig(ROOT/'figures/hk-redox-update/partial_fit.png',dpi=160);plt.close(fig)
    psd=np.load(OUT/'llnv_saved_voltage_psd.npz');f=psd['frequency_hz'];sel=(f>=1)&(f<=1000)
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    for group,color in [('WTCRY','#2b6d9c'),('CryNull','#bd7548')]:
        for tag,style in [('reference','-'),('earlier_validation','--')]:
            ax.loglog(f[sel],psd[group+'__'+tag+'__file_equal_mean'][sel],style,color=color,label=f'{group}: {tag}')
    ax.set(xlabel='Frequency (Hz)',ylabel='Saved Vm PSD (mV^2 / Hz)',
           title='l-LNv Blue files: AP-including baseline fluctuations')
    ax.legend(fontsize=8);fig.text(.5,.01,'No analog filter correction; intrinsic noise components not separated.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(ROOT/'figures/hk-redox-update/llnv_psd.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
