"""Review-directed broad baseline sensitivity, pre-labelled as follow-up."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl import synaptic_veto as v
OUT=ROOT/'data/synaptic-veto'
rows=[]
for offset in [-1.5,1.5]:
 for route in v.ROUTES:
  for seed in [26091711,26091712,26091713]:
   c=v.Config(gain=1024,tau=.2,route=route,offset=offset)
   r=v.simulate(c,n=1024,seed=seed)
   for row in v.summarize(r):row['followup']='far_background';rows.append(row)
   np.savez_compressed(OUT/'trials'/f'far_background_{offset}_{route}_{seed}.npz',times=r['times'],outcomes=r['outcomes'])
pd.DataFrame(rows).to_csv(OUT/'far_background.csv',index=False)
print('far background done')
