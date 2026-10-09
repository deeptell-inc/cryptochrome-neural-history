from pathlib import Path
import sys,json,time,argparse
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl.hq_rotation import frozen
from impl.hq_rotation_cases import CaseCycle,storage_curve
p=argparse.ArgumentParser();p.add_argument('--cutoff',type=int,default=2);p.add_argument('--validation-only',action='store_true');args=p.parse_args()
D=ROOT/'data/rotation-cases';D.mkdir(exist_ok=True)
with threadpool_limits(limits=1):
 c=CaseCycle(frozen(),args.cutoff);results=[]
 specs=[('free',1.,False,()),('protect_HQ',1.,False,('H',)),('protect_waits',1.,False,('H','R','E'))]
 specs += [(f'all_fast_{m:g}',m,False,()) for m in [1e2,1e4,1e6,1e8]]
 specs += [(f'HQ_fast_{m:g}',m,True,()) for m in [1e4,1e8]]
 if args.validation_only:specs=[s for s in specs if s[0] in ['protect_HQ','protect_waits','all_fast_10000','all_fast_1e+06','all_fast_1e+08']]
 for label,m,hq,protected in specs:
  start=time.monotonic();x,result=c.run_case(label,m,hq,protected)
  if label in ['all_fast_10000','all_fast_1e+06','all_fast_1e+08']:
   result['extra_HQ_delay']=storage_curve(c,x,[0.,1e-6,1e-5,1e-4,.001,.012])
  result['seconds']=time.monotonic()-start
  np.savez_compressed(D/f'molecular_L{args.cutoff}_{label}.npz',state=x)
  (D/f'molecular_L{args.cutoff}_{label}.json').write_text(json.dumps(result,indent=2)+'\n')
  results.append(result);print(json.dumps(result),flush=True)
 (D/f'molecular_L{args.cutoff}_all.json').write_text(json.dumps(results,indent=2)+'\n')
