from pathlib import Path
import sys,os,json,time,argparse
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from threadpoolctl import threadpool_limits
from impl.hq_rotation import *
p=argparse.ArgumentParser();p.add_argument('--cutoff',type=int,default=1);p.add_argument('--tau-ns',type=float,default=21.476595751657293);p.add_argument('--field-uT',type=float,default=50.);p.add_argument('--check-aligned',action='store_true');a=p.parse_args()
with threadpool_limits(limits=1):
 start=time.monotonic();f=frozen();cycle=Cycle(f,a.cutoff,a.tau_ns*1e-9,a.field_uT*1e-6);x,result=cycle.run()
 if a.check_aligned:
  _,aligned_result=cycle.run(aligned=True);result['aligned_control']=aligned_result
 result.update(cutoff=a.cutoff,tau_ns=a.tau_ns,field_uT=a.field_uT,seconds=time.monotonic()-start)
 tag=f'L{a.cutoff}_tau{a.tau_ns:.8g}_B{a.field_uT:g}';out=ROOT/'data/hq-rotation';np.savez_compressed(out/(tag+'.npz'),state=x);(out/(tag+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
