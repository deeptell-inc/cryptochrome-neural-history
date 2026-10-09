from pathlib import Path
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from threadpoolctl import threadpool_limits
from impl.hq_rotation import frozen
from impl.hq_electric import ElectricCycle
p=argparse.ArgumentParser();p.add_argument('--L',type=int,default=2);p.add_argument('--kappa',type=float,default=10.910941626120245);p.add_argument('--axis',type=int,default=2);a=p.parse_args()
with threadpool_limits(limits=1):
 t=time.monotonic();c=ElectricCycle(frozen(),a.L,a.kappa,a.axis);x,r=c.run();r.update(kappa=a.kappa,cutoff=a.L,axis=a.axis,seconds=time.monotonic()-t)
 out=ROOT/'data/membrane-field'/f'free_L{a.L}_k{a.kappa:.8g}_axis{a.axis}.json';out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
