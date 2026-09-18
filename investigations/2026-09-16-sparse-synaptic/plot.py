from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'data/sparse-synaptic'
d=pd.read_csv(OUT/'neural_aggregate.csv');r=pd.read_csv(OUT/'race_aggregate.csv')
arms=['diffuse','random_sparse','targeted','weak','opposite'];colors=['#64748b','#ca8a04','#0f766e','#7c3aed','#be123c']
fig,axs=plt.subplots(2,2,figsize=(12,8.5),layout='constrained')
for offset,route,color in [(-.25,'go','#2563eb'),(0,'trigger','#ea580c'),(.25,'stop','#0f766e')]:
    z=d.query('run=="primary" and gain_probability_per_yield==512 and route==@route').set_index('arm')
    axs[0,0].bar(np.arange(5)+offset,z.loc[arms].delta_RMST_s*1000,width=.24,label=route,color=color)
axs[0,0].axhline(0,c='black',lw=.6);axs[0,0].set_xticks(np.arange(5),['Diffuse','Random8','Target8','Weak8','Opposite8'],rotation=15)
axs[0,0].set_ylabel('Change in restricted mean time (ms)');axs[0,0].legend(frameon=False)
axs[0,0].set_title('A  Equal total probability change; different locations')
for route,color in [('go','#2563eb'),('trigger','#ea580c'),('stop','#0f766e')]:
    z=r.query('run=="primary" and gain_probability_per_yield==512 and route==@route and arm=="targeted"')
    axs[0,1].plot(z.SSD_s,z.delta_response*100,'o-',label=route,color=color)
axs[0,1].axhline(0,c='black',lw=.6);axs[0,1].set_xlabel('Stop-signal delay (s)');axs[0,1].set_ylabel('Change in response probability (pp)')
axs[0,1].set_title('B  Action site changes the stop-task outcome');axs[0,1].legend(frameon=False)
traces=[]
for seed in (26091691,26091692,26091693):
    z=np.load(OUT/f'primary_{seed}_512_go_trials.npz');traces.append(z['mean_state'])
mean=np.mean(traces,axis=0);t=(np.arange(len(mean))+1)*.002
for arm,color in zip(arms,colors):
    j=['baseline','diffuse','random_sparse','targeted','weak','opposite'].index(arm)
    axs[1,0].plot(t,mean[:,j]-mean[:,0],color=color,label=arm)
axs[1,0].axvline(.203,c='black',ls=':',lw=.8);axs[1,0].set_xlim(.15,.9)
axs[1,0].set_xlabel('Time from go-task onset (s)');axs[1,0].set_ylabel('Change in preparation state P (dimensionless)')
axs[1,0].set_title('C  Cue-aligned state, all synthetic trials');axs[1,0].legend(frameon=False,fontsize=8)
proxies=json.loads((OUT/'readiness_proxies.json').read_text())
for arm,color in [('targeted','#0f766e'),('weak','#7c3aed')]:
    rows=[next(x for x in p['records'] if x['arm']==arm) for p in proxies if p['run']=='primary' and p['gain']==512]
    tt=rows[0]['time_s']
    own=np.mean([np.array(x['common_own'])-x['common_baseline'] for x in rows],0)
    fixed=np.mean([np.array(x['common_fixed'])-x['common_baseline'] for x in rows],0)
    axs[1,1].plot(tt,own,color=color,label=arm+' / own movement')
    axs[1,1].plot(tt,fixed,color=color,ls='--',label=arm+' / baseline movement')
axs[1,1].set_xlabel('Aligned time (s)');axs[1,1].set_ylabel('Change in normalized -P proxy')
axs[1,1].set_title('D  Matched trial cohort; alignment control');axs[1,1].legend(frameon=False,fontsize=8)
for ax in axs.flat:
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
fig.suptitle('Conditional synthetic circuit: selectivity is possible, quantum origin is not identified\nUncalibrated gain 512 / yield; 12 ms signal; no EEG or SSRT calibration',fontsize=12)
fig.savefig(OUT/'summary.png',dpi=170);fig.savefig(OUT/'summary.svg');plt.close(fig)
