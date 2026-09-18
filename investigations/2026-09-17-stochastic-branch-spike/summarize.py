from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from scipy.stats import norm
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'data/stochastic-branch-spike';config=json.loads((OUT/'configuration.json').read_text())
def first(response,tag):return np.concatenate([np.load(OUT/f'{response}__{tag}__seed{seed}.npz')['first_s'] for seed in config['seeds']],axis=1)
summary=pd.read_csv(OUT/'pooled_summary.csv');power=[];ablation=[]
for response in ['linear_gate','bounded_amplifier']:
    row=summary[(summary.response==response)&(summary['case']=='both')].iloc[0]
    p0=row.reference_probability;p1=row.history_probability;pm=(p0+p1)/2;delta=p1-p0
    for comparisons in [1,2]:
        n=((norm.ppf(1-.05/(2*comparisons))*np.sqrt(2*pm*(1-pm))+norm.ppf(.8)*np.sqrt(p0*(1-p0)+p1*(1-p1)))/delta)**2
        power.append(dict(response=response,comparisons=comparisons,alpha_family=.05,power=.8,
           independent_Bernoulli_trials_per_arm=int(np.ceil(n)),scope='normal approximation; ignores between-cell heterogeneity, correlation, nuisance calibration and repeated testing'))
    control=first(response,'background_only');dc=np.isfinite(control[1]).astype(float)-np.isfinite(control[0])
    for tag in ['molecule_only','channel_only','both']:
        tr=first(response,tag);d=np.isfinite(tr[1]).astype(float)-np.isfinite(tr[0]);diff=d-dc
        ablation.append(dict(response=response,case=tag,delta_history_effect_vs_background=float(diff.mean()),
           shared_MC_SE=float(diff.std(ddof=1)/np.sqrt(len(diff))),interpretation='numerical common-noise ablation; contributions not additive in nonlinear cascade'))
pd.DataFrame(power).to_csv(OUT/'independent_trial_power_design.csv',index=False)
pd.DataFrame(ablation).to_csv(OUT/'noise_ablation_comparison.csv',index=False)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(12,8));grid=np.linspace(0,.05,201)
truth=first('bounded_amplifier','both');prediction=first('bounded_amplifier','upgraded_prediction')
for arm,label,color in [(0,'Reference','#426e86'),(1,'History','#bc4c35')]:
    axes[0,0].plot(grid*1000,np.mean(truth[arm,:,None]<=grid,axis=0),label=label+' truth',color=color)
    axes[0,0].plot(grid*1000,np.mean(prediction[arm,:,None]<=grid,axis=0),'--',label=label+' calibrated prediction',color=color)
axes[0,0].set(xlabel='Time (ms)',ylabel='P(first spike by time)',title='Sensitive relay: held-out distributions');axes[0,0].legend(fontsize=8)
for response,color in [('linear_gate','#426e86'),('bounded_amplifier','#bc4c35')]:
    d=summary[(summary.response==response)&summary['case'].isin(['background_only','molecule_only','channel_only','both'])].set_index('case').loc[['background_only','molecule_only','channel_only','both']]
    axes[0,1].errorbar(np.arange(4),100*d.delta_probability,yerr=196*d.paired_MC_SE,marker='o',capsize=3,label=response,color=color)
axes[0,1].set(xticks=np.arange(4),xticklabels=['Background','+ molecule','+ channel','Both'],ylabel='History change (percentage points)',title='95% Monte Carlo error bars, not biological CI');axes[0,1].legend(fontsize=8)
bins=np.linspace(0,.05,26)
for arm,label,color in [(0,'Reference','#426e86'),(1,'History','#bc4c35')]:
    density=np.histogram(truth[arm,np.isfinite(truth[arm])],bins=bins)[0]/truth.shape[1]/(np.diff(bins)*1000)
    axes[1,0].step((bins[:-1]+bins[1:])*500,density,where='mid',label=f'{label}: no spike {100*np.mean(~np.isfinite(truth[arm])):.1f}%',color=color)
axes[1,0].set(xlabel='First-spike time (ms)',ylabel='Unconditional density / ms',title='Censoring retained, no artificial 50 ms spike');axes[1,0].legend(fontsize=8)
initial=pd.read_csv(OUT/'heldout_prediction_checks.csv');up=pd.read_csv(OUT/'upgraded_prediction_checks.csv')
axes[1,1].plot(100*initial.heldout_CDF_max_error.to_numpy(),'o-',label='Initial constant-step calibration')
axes[1,1].plot(100*up.CDF_max_error.to_numpy(),'o-',label='Independent pulse calibration')
axes[1,1].axhline(100*up.DKW_bound.iloc[0],ls='--',color='gray',label='Conservative 12-comparison MC bound')
axes[1,1].set(xlabel='Held-out case (relay, seed, arm)',ylabel='Maximum CDF gap (percentage points)',title='Membrane calibration was the initial bottleneck');axes[1,1].legend(fontsize=7)
fig.tight_layout();fig.savefig(OUT/'firing_distribution.png',dpi=170);fig.savefig(OUT/'firing_distribution.svg');plt.close(fig)
stats=dict(initial_max_CDF_error=float(initial.heldout_CDF_max_error.max()),upgraded_max_CDF_error=float(up.CDF_max_error.max()),
    initial_failures=int((~initial.within_DKW).sum()),upgraded_failures=int((~up.within_DKW).sum()),
    caveat='DKW bounds Monte Carlo sampling under equal distributions; failure to reject does not establish equivalence or native calibration')
(OUT/'summary.json').write_text(json.dumps(stats,indent=2)+'\n')
print(pd.DataFrame(power).to_string(index=False));print(json.dumps(stats,indent=2))
