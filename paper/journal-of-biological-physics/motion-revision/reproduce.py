"""Bounded reproduction: frozen result arithmetic, intervals, tables and figures.
Does not repeat upstream electronic structure, MD or stochastic neural simulation.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import beta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext
P=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','axes.labelsize':9,'legend.fontsize':8})
C=['#0072B2','#D55E00','#009E73','#CC79A7']
def read(d,n):return pd.read_csv(P/'data'/d/n)
def sci(x):
 a,b=f'{x:.5e}'.split('e');return rf'${a}\times10^{{{int(b)}}}$'
def table(name,rows): (P/'tables'/name).write_text('\n'.join(' & '.join(map(str,r))+r' \\' for r in rows)+'\n')
def save(fig,n):
 fig.savefig(P/f'Fig{n}.pdf',bbox_inches='tight');fig.savefig(P/f'Fig{n}.svg',bbox_inches='tight');fig.savefig(P/'audit'/f'Fig{n}.png',dpi=160,bbox_inches='tight');plt.close(fig)
checks={}
manifest=json.loads((P/'audit/input-manifest.json').read_text())
checks['frozen_inputs_sha256']=all(hashlib.sha256((P/x['path']).read_bytes()).hexdigest()==x['sha256'] for x in manifest)
f=read('hq-rotation','summary.csv');r=read('rotation-cases','molecular_summary.csv');b=read('binding-motion','molecular_summary.csv');n=read('binding-motion','neural_selected.csv');e=read('field-couplings','cycle_summary.csv');v=read('field-couplings','electronic_summary.csv');m=read('membrane-field','summary.csv')
checks['free_unresolved']=bool(np.max(abs(f.full-f.reset))<1e-10)
checks['rotation_delta']=bool(np.max(abs(r.full-r.reset-r.delta))<1e-12)
checks['binding_decomposition']=bool(np.max(abs(b.spin_contribution+b.motion_contribution-b.delta_q))<1e-11)
checks['electric_delta']=bool(np.max(abs(e.q_full-e.q_reset-e.delta_yield))<1e-12)
errs=[]
for row in n.itertuples():
 nn=row.paired_trials
 def ci(k): return (0. if not k else beta.ppf(.0125,k,nn-k+1),1. if k==nn else beta.ppf(.9875,k+1,nn-k))
 pl,pu=ci(row.plus);ml,mu=ci(row.minus)
 errs.extend([abs(row.delta_stop_pp-100*(row.plus-row.minus)/nn),abs(row.low_pp-100*(pl-mu)),abs(row.high_pp-100*(pu-ml))])
checks['neural_counts_and_intervals']=bool(max(errs)<1e-10)
checks['maximum_interval_arithmetic_error']=max(errs)
# highest available angular cutoff for each physical binding condition
cols=['angle_deg','tau_w_ns','target_bound_fraction','koff_s','bias']
best=b.sort_values('cutoff').drop_duplicates(cols,keep='last')
def bind(angle=1.,fraction=.5,off=1.,tau=.01,bias=0.):
 return best[(best.angle_deg==angle)&(best.target_bound_fraction==fraction)&(best.koff_s==off)&(best.tau_w_ns==tau)&(best.bias==bias)].iloc[0]
# Fig1 spectrum and counterfactual interventions
spec=pd.DataFrame(json.loads((P/'data/hq-rotation/HQ_spectrum.json').read_text())).sort_values('cutoff').drop_duplicates('tau_ns',keep='last').sort_values('tau_ns')
fig,ax=plt.subplots(1,2,figsize=(7,3.0),layout='constrained')
ax[0].plot(spec.tau_ns,spec.inverse_slowest_rate_us,'o-',c=C[0]);ax[0].set(xscale='log',xlabel=r'Rotational correlation time $\tau_2$ (ns)',ylabel=r'Inverse slowest HQ decay rate ($\mu$s)',title='a  Free rotational relaxation')
names=['protect_HQ','protect_waits','all_fast_10000','all_fast_1e+06','all_fast_1e+08'];lab=['HQ protected','All waits protected',r'All waits $\times 10^4$',r'All waits $\times 10^6$',r'All waits $\times 10^8$']
y=[max(1e-10,float(r[r.case==k].delta.iloc[0])) for k in names]
ax[1].barh(lab,y,color=[C[0],C[1],C[2],C[2],C[2]],height=.65)
ax[1].set(xscale='log',xlim=(5e-11,7e-4),xlabel=r'History contrast $\Delta q$',title='b  Protection and rapid renewal');ax[1].invert_yaxis();ax[1].axvline(1e-10,ls=':',color='k',lw=.8);ax[1].annotate('unresolved',xy=(1.5e-10,0),fontsize=8)
save(fig,1)
# Fig2 reversible binding -- avoid artificial zero wobble log axes
fig,ax=plt.subplots(1,2,figsize=(7,3.1),layout='constrained')
for j,angle in enumerate([0.,.1,1.,5.]):
 off=np.array([100.,1.,.01]);vals=[bind(angle=angle,off=x).delta_q for x in off]
 ax[0].plot(1/off,vals,['o-','s--','^-.','D:'][j],color=C[j],label=f'{angle:.5f}°')
ax[0].set(xscale='log',yscale='log',xlabel='Mean bound dwell time (s)',ylabel=r'History contrast $\Delta q$',title='a  Wobble and dwell time');ax[0].legend(title='RMS rotation angle',frameon=False)
for j,off in enumerate([.01,1.,100.]):
 fs=np.array([.01,.1,.5,.9]);ax[1].plot(fs,[bind(fraction=x,off=off).delta_q for x in fs],['o-','s--','^:'][j],color=C[j],label=f'{1/off:.5f} s')
ax[1].set(yscale='log',xlabel='Motion-only equilibrium bound fraction',ylabel=r'History contrast $\Delta q$',title='b  Bound fraction at 1.00000°');ax[1].legend(frameon=False)
save(fig,2)
# Fig3 electronic and mechanical factors at same imposed orientation
fig,ax=plt.subplots(1,2,figsize=(7,3.05),layout='constrained')
ax[0].bar(np.arange(len(v)),100*v.EFG_max_relative_change_at_14MVm,color=[C[0]]*2+[C[1]]*2+[C[2]]*2)
ax[0].set_xticks(np.arange(len(v)),[f'{x.state}/{x.atom}' for x in v.itertuples()],rotation=30);ax[0].set(ylabel='Maximum linear EFG change (%)',title='a  Frozen-geometry electronic response')
for j,angle in enumerate([1.,5.]):
 rows=e[(e.cutoff==2)&(e.rms_deg==angle)].set_index('mode');baseline=rows.loc['baseline','delta_yield'];values=[100*(rows.loc[k,'delta_yield']/baseline-1) for k in ['electronic','well','combined']]
 ax[1].bar(np.arange(3)+(j-.5)*.36,values,.36,color=C[j],hatch=['','///'][j],label=f'{angle:.5f}°')
ax[1].axhline(0,color='k',lw=.6);ax[1].set_xticks(range(3),['Electronic','Well','Both']);ax[1].set(ylabel=r'Change in $\Delta q$ from orientation only (%)',title='b  Separate contributions at fixed κ');ax[1].legend(title='Initial RMS angle',frameon=False)
save(fig,3)
# Fig4 selected conditional readouts with all counts retained in source CSV
keys=['angle0_tau1e-11_f0.5_off1_bias0','angle1_tau1e-11_f0.5_off1_bias0','angle5_tau1e-11_f0.5_off1_bias0','angle1_tau1e-11_f0.5_off100_bias0','angle1_tau1e-11_f0.01_off1_bias0','angle0_tau1e-11_f0_off0_bias0']
sel=n.set_index('label').loc[keys].reset_index()
labels=['0.00000°, 1.00000 s, f=0.50000','1.00000°, 1.00000 s, f=0.50000','5.00000°, 1.00000 s, f=0.50000','1.00000°, 0.01000 s, f=0.50000','1.00000°, 1.00000 s, f=0.01000','Free rotation / equal input']
fig,ax=plt.subplots(figsize=(7,3.1),layout='constrained');val=sel.delta_stop_pp.to_numpy();err=np.array([val-sel.low_pp,sel.high_pp-val]);ax.errorbar(val,np.arange(6),xerr=err,fmt='o',color=C[0],capsize=3);ax.set_yticks(range(6),labels);ax.invert_yaxis();ax.axvline(0,c='k',lw=.8,ls=':');ax.set(xlabel='Conditional change in stopping probability (percentage points)',title='Assumed gain 1024; 24,576 paired trials per condition');save(fig,4)
# all article numbers derive from these full-precision summaries
table('free.tex',[[f'{x.tau_ns:.5f}',f'{x.full:.5f}',f'{x.reset:.5f}',f'{spec.iloc[i].inverse_slowest_rate_us:.5f}',r'$<10^{-10}$'] for i,x in enumerate(f.itertuples())])
table('binding.tex',[[f'{angle:.5f}']+[sci(bind(angle=angle,off=off).delta_q) for off in [.01,1.,100.]] for angle in [0.,.1,1.,5.]])
rows=[]
for angle in [1.,5.]:
 sub=e[(e.cutoff==2)&(e.rms_deg==angle)].set_index('mode');base=sub.loc['baseline','delta_yield']
 for mode,label in [('baseline','Orientation only'),('electronic','Electronic'),('well','Well'),('combined','Both')]:
  row=sub.loc[mode];rows.append([f'{angle:.5f}',label,sci(row.delta_yield),f'{100*(row.delta_yield/base-1):.5f}'])
table('electric.tex',rows)
table('electronic.tex',[[x.state,x.atom,f'{100*x.EFG_max_relative_change_at_14MVm:.5f}',f'{x.isotropic_HFC_max_change_MHz_at_14MVm:.5f}' if x.state=='E' else '--',f'{x.baseline_isotropic_HFC_MHz:.5f}' if x.state=='E' else '--'] for x in v.itertuples()])
table('neural.tex',[[str(i+1),f'{x.delta_stop_pp:.5f}',f'[{x.low_pp:.5f}, {x.high_pp:.5f}]',str(x.plus),str(x.minus)] for i,x in enumerate(sel.itertuples())])
table('fast.tex',[[x.case.replace('all_fast_','').replace('1e+06',r'$10^6$').replace('1e+08',r'$10^8$').replace('10000',r'$10^4$'),sci(x.delta),sci(x.full_population_gap),f'{x.mean_preparation_time_s*1e6:.5f}'] for x in r.itertuples() if x.case in ['all_fast_10000','all_fast_1e+06','all_fast_1e+08']])
checks['all_boolean_checks_passed']=all(x for x in checks.values() if isinstance(x,bool))
(P/'audit/reproduction-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
assert checks['all_boolean_checks_passed'],checks
print(json.dumps(checks,indent=2))
