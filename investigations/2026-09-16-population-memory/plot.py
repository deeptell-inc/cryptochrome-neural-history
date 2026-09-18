from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/population-memory'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
m=pd.read_csv(P/'write_read_modes.csv').query('stage=="after_20_cycles" and gamma_oxygen_s==0').copy()
colors=np.where(m.correlation_fraction>.99,'#c47b33',np.where(m.N5_local_fraction>.99,'#4277a5','#4b9876'))
ax[0,0].bar(m['mode'],100*m.contribution_at_zero/m.contribution_at_zero.sum(),color=colors)
ax[0,0].set(xlabel='HQ population relaxation mode',ylabel='Share of yield-memory signal (%)',title='Write + read: four nuclear-correlation modes matter')
from matplotlib.patches import Patch
ax[0,0].legend(handles=[Patch(color='#4277a5',label='N5 local'),Patch(color='#4b9876',label='N10 local'),Patch(color='#c47b33',label='N5--N10 correlation')],fontsize=8)
h=pd.read_csv(P/'HQ_pulse_chase.csv').query('time_s>=.001')
initial=h.population_yield_memory.iloc[0]
ax[0,1].semilogx(h.time_s,h.full_yield_memory/initial,label='Full',lw=3)
ax[0,1].semilogx(h.time_s,h.population_yield_memory/initial,'--',label='8 population modes',lw=2)
ax[0,1].axvline(1.63474,color='gray',ls=':',label='Half-signal: 1.63 s (conditional)')
ax[0,1].set(xlabel='Additional hold in HQ (s)',ylabel='Readable memory / initial',title='Isolated HQ hold: population relaxation is sufficient');ax[0,1].legend(fontsize=8)
r=pd.read_csv(P/'reaction_renewal_memory.csv')
for axis,value,label in [('baseline',1,'Baseline'),('h_noise_scale',0,'HQ dissipator off'),('escape_noise_scale',0,'E dissipator off')]:
    d=r.query('axis==@axis and value==@value');ax[1,0].plot(d.additional_cycles,d.relative_memory,marker='o',label=label)
d=pd.read_csv(P/'no_bath_reaction_memory.csv');ax[1,0].plot(d.additional_cycles,d.relative_memory,'--o',label='All environmental dissipators off')
ax[1,0].set(xlabel='Further completed reaction cycles',ylabel='Pre-existing readable memory / initial',title='Cycling: recovery and reactive maps also erase memory');ax[1,0].legend(fontsize=8)
p=pd.read_csv(P/'phase_only_control.csv')
ax[1,1].plot(p.time_s*1e6,p.coherence_yield_part*1e6,label='Fixed coherent waiting time')
ax[1,1].axhline(0,color='#c47b33',ls='--',label='Phase-averaged coherence contribution')
ax[1,1].set(xlabel='HQ hold after one recovery (microseconds)',ylabel='Coherence contribution to yield (x 1e-6)',title='Phase averaging: coherence cancels, populations survive');ax[1,1].legend(fontsize=8)
fig.suptitle('Nuclear population memory in the current dark-CRY model\nConditional tensors, noise spectrum and kinetics; no native T1 calibration',fontsize=14)
fig.savefig(P/'population_memory.png',dpi=160);fig.savefig(P/'population_memory.svg');plt.close(fig)
