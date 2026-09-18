"""Review-driven controls: no detector, literal majority, nonmaintained product."""
from pathlib import Path
from dataclasses import replace
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl import synaptic_veto as v
OUT=ROOT/'data/synaptic-veto'

def main():
 rows=[]
 for label in ['no_detector','literal_majority','depletion']:
  for route in v.ROUTES:
   for seed in [26091711,26091712,26091713]:
    c=v.Config(gain=1024,tau=.2,route=route,heterogeneity=0. if label=='literal_majority' else .15)
    r=v.simulate(c,n=2048,seed=seed,detector_enabled=label!='no_detector',recovery='depletion' if label=='depletion' else 'stationary')
    for row in v.summarize(r):row['supplement']=label;rows.append(row)
    np.savez_compressed(OUT/'trials'/f'supplement_{label}_{route}_{seed}.npz',times=r['times'],outcomes=r['outcomes'])
  print(label,'done',flush=True)
 pd.DataFrame(rows).to_csv(OUT/'supplement.csv',index=False)
if __name__=='__main__':main()
