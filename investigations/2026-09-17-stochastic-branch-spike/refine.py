"""Targeted high-count refinement of the sensitive relay timestep comparison."""
from pathlib import Path
import sys,json
from dataclasses import replace
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl import stochastic_branch_spike as s
from impl import branch_current_gain as b
OUT=ROOT/'data/stochastic-branch-spike';c=json.loads((OUT/'configuration.json').read_text())
y0=c['reference_yield'];dy=c['history_delta_yield'];p=b.Scenario(channel_response='bounded_amplifier',capacity_nS=3.125,tau_gate_s=.005)
allfirst=[];rows=[]
for seed in c['seeds']:
    r=s.simulate_pair(y0,dy,p,n=16384,seed=seed+90000,dt=.00005)
    np.savez_compressed(OUT/f'bounded_amplifier__high_count_fine__seed{seed}.npz',**r)
    allfirst.append(r['first_s']);rows.append(dict(seed=seed,**s.contrast(r['first_s'])))
    print(rows[-1],flush=True)
first=np.concatenate(allfirst,axis=1);base=s.distribution(first[0]);hist=s.distribution(first[1]);delta=s.contrast(first)
summary=dict(n_per_arm=first.shape[1],dt_s=.00005,reference=base,history=hist,contrast=delta,
    role='targeted audit; primary main run preserved; Monte Carlo uncertainty distinct from calibration uncertainty')
(OUT/'high_count_refinement.json').write_text(json.dumps(summary,indent=2)+'\n')
pd.DataFrame(rows).to_csv(OUT/'high_count_refinement_by_seed.csv',index=False)
