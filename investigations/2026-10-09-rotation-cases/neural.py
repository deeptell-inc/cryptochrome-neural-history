"""Scenario sources injected without altering the archived neural implementation."""
from pathlib import Path
import sys,json,time,argparse
from dataclasses import replace
from unittest.mock import patch
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl import synaptic_veto as v
BASE=ROOT/'data/rotation-cases';p=argparse.ArgumentParser();p.add_argument('--batch',choices=['mixture','routes','fast','field'],default='mixture');p.add_argument('--n',type=int,default=4096);p.add_argument('--output-subdir',default='reference-runtime');args=p.parse_args()
D=BASE/args.output_subdir;D.mkdir(exist_ok=True)
runtime=dict(python=sys.version,numpy=np.__version__,source='archived neural runtime')
if (D/'runtime.json').exists():
 old_runtime=json.loads((D/'runtime.json').read_text());assert old_runtime['numpy']==runtime['numpy'] and old_runtime['python']==runtime['python'],'Use a new output subdirectory for a different runtime.'
(D/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
q0,ys=v.sources();q1=ys['full'];qfree=json.loads((ROOT/'data/hq-rotation/independent_L4_tau21.476596.json').read_text())['yield_reset_uniform']
scenarios=[]
if args.batch=='mixture':
 for f in [0.,.001,.01,.1,.5,1.]:
  ref=f*q0+(1-f)*qfree;src=ref+f*(q1-q0)
  scenarios.append((f'protected_fraction_{f:g}',ref,src,'stop',.2,('top','diffuse','opposite')))
elif args.batch=='routes':
 for mode,ref,src in [('clamped',q0,q1),('free_null',qfree,qfree)]:
  for route in v.ROUTES:
   for tau in [.012,.2]:scenarios.append((mode,ref,src,route,tau,('top',)))
elif args.batch=='field':
 ref=json.loads((ROOT/'data/hq-rotation/L0_tau21.476596_B0.json').read_text())['yields']['reset']
 for route in v.ROUTES:
  for tau in [.012,.2]:scenarios.append(('field_0_to_50uT',ref,qfree,route,tau,('top',)))
else:
 for label in ['protect_waits','all_fast_10000','all_fast_1e+06','all_fast_1e+08']:
  molecular=json.loads((BASE/f'molecular_L3_{label}.json').read_text())
  ref=molecular['yields']['reset'];src=molecular['yields']['full']
  for tau in [.012,.2]:scenarios.append((label,ref,src,'stop',tau,('top',)))
  if label in ['all_fast_1e+06','all_fast_1e+08']:
   for tau in [.012,.2]:scenarios.append((label+'_population',ref,molecular['yields']['population'],'stop',tau,('top',)))
SEEDS=[26091711,26091712,26091713]
plan=[dict(label=l,q0=q,q1=r,route=route,tau_s=t,n=args.n,seeds=SEEDS,names=names) for l,q,r,route,t,names in scenarios]
(D/f'neural_plan_{args.batch}.json').write_text(json.dumps(plan,indent=2)+'\n');rows=[]
with threadpool_limits(limits=1):
 for index,(label,ref,src,route,tau,names) in enumerate(scenarios):
  for seed in SEEDS:
   tag=f'{args.batch}_{index}_{seed}';out=D/f'neural_{tag}.csv'
   if out.exists():
    cached=pd.read_csv(out)
    assert (cached['n']==args.n).all() and np.allclose(cached.q0,ref,rtol=0,atol=1e-15) and np.allclose(cached.q1,src,rtol=0,atol=1e-15),'Cached input mismatch; use a new output subdirectory.'
    rows.extend(cached.to_dict('records'));continue
   c=replace(v.Config(),gain=1024.,route=route,tau=tau)
   start=time.monotonic()
   with patch.object(v,'sources',return_value=(ref,{'full':src,'population':src,'reset':ref})):
    result=v.simulate(c,n=args.n,seed=seed,names=names)
   if ref==src:assert np.array_equal(result['outcomes'][:,:,0],result['outcomes'][:,:,1])
   block=v.summarize(result)
   for row in block:row.update(scenario=label,q0=ref,q1=src,delta_q=src-ref,reporting_resolution_free_delta=1e-10 if ref==src else None)
   pd.DataFrame(block).to_csv(out,index=False);rows.extend(block)
   np.savez_compressed(D/f'neural_{tag}.npz',outcomes=result['outcomes'],times=result['times'])
   print(json.dumps(dict(label=label,route=route,tau=tau,seed=seed,n=args.n,seconds=time.monotonic()-start)),flush=True)
 pd.DataFrame(rows).to_csv(D/f'neural_{args.batch}_all.csv',index=False)
