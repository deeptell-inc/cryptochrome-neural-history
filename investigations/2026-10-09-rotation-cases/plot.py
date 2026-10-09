from pathlib import Path
import json
import numpy as np,pandas as pd
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/hq-rotation-cases-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/private/tmp/hq-rotation-cases-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'data/rotation-cases'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
d=pd.read_csv(D/'neural_selected.csv');fig,ax=plt.subplots(1,3,figsize=(14,4.6),constrained_layout=True)
m=d.query('batch=="mixture"').copy();m['f']=m.scenario.str.replace('protected_fraction_','').astype(float);m=m.sort_values('f')
ax[0].errorbar(m.f,m.delta_stop_pp,yerr=[m.delta_stop_pp-m.conservative_low_pp,m.conservative_high_pp-m.delta_stop_pp],fmt='o-',color='#27617a',capsize=3)
ax[0].set_xscale('symlog',linthresh=.001);ax[0].set_xlabel('Immobile fraction f (two populations)');ax[0].set_ylabel('Change in stopping probability (pp)');ax[0].set_title('(a) Protected subpopulation');ax[0].axhline(0,color='gray',lw=.7)
labels=['protect_waits','all_fast_1e+06','all_fast_1e+08','field_0_to_50uT'];names=['Protected\nwaiting states','Waiting rates\n'+r'$\times10^6$','Waiting rates\n'+r'$\times10^8$','Single probe\n0 vs 50 '+r'$\mu$T']
z=d[(d.scenario.isin(labels))&(d.route=='stop')&(d.tau_s==.2)].set_index('scenario').loc[labels]
x=np.arange(len(z));ax[1].bar(x,z.delta_stop_pp,color=['#599b7d','#beaa4a','#beaa4a','#7063a5'],alpha=.9)
ax[1].errorbar(x,z.delta_stop_pp,yerr=[z.delta_stop_pp-z.conservative_low_pp,z.conservative_high_pp-z.delta_stop_pp],fmt='none',ecolor='black',capsize=4)
ax[1].set_xticks(x,names);ax[1].set_ylabel('Change in stopping probability (pp)');ax[1].set_title('(b) Conditional alternative cases');ax[1].axhline(0,color='gray',lw=.7)
for label,color in [('all_fast_1e+06','#beaa4a'),('all_fast_1e+08','#ad604e')]:
 r=json.loads((D/f'molecular_L3_{label}.json').read_text());p=pd.DataFrame(r['extra_HQ_delay']);p=p[p.delay_s>0];ax[2].plot(p.delay_s,np.maximum(abs(p.delta_yield),1e-12),'o-',label=label.replace('all_fast_','rates x '),color=color)
p=pd.DataFrame(json.loads((D/'protected_wait_storage.json').read_text())['extra_delay']);p=p[p.delay_s>0];ax[2].plot(p.delay_s,p.delta_yield,'s-',label='protected waiting states',color='#599b7d')
ax[2].set_xscale('log');ax[2].set_yscale('log');ax[2].set_ylim(1e-12,1e-3);ax[2].axhline(1e-10,color='gray',linestyle='--',lw=.8);ax[2].set_xlabel('Extra HQ storage delay (s)');ax[2].set_ylabel('Absolute reaction-yield contrast');ax[2].set_title('(c) Immediate signal vs retained history');ax[2].legend(fontsize=8,loc='lower left')
fig.suptitle('Conditional model: 24,576 paired trials; chemical lifetime 200 ms; gain 1024\nIntervals in (a,b): conservative paired 95% Monte Carlo intervals; no biological calibration',fontsize=11)
fig.savefig(D/'case_summary.png',dpi=180);fig.savefig(D/'case_summary.pdf');plt.close(fig)
