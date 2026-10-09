from pathlib import Path
import sys,json,argparse,time,gc
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl.hq_rotation import frozen
from impl.hq_binding import BindingCycle
p=argparse.ArgumentParser();p.add_argument('--cutoff',type=int,default=3);p.add_argument('--angles',nargs='+',type=float,default=[0.,5.]);p.add_argument('--tau-w',type=float,default=1e-11);args=p.parse_args()
D=ROOT/'data/binding-motion'
with threadpool_limits(limits=1):
 f=frozen();previous=None
 for angle in args.angles:
  c=BindingCycle(f,args.cutoff,rms_deg=angle,tau_w=args.tau_w,mobile_reference=previous);previous=c;gc.collect()
  for fraction,off,bias in [(.5,.01,0.),(.5,1.,0.),(.5,100.,0.),(.5,1.,1.8),(.5,0.,0.)]:
   label=f'L{args.cutoff}_angle{angle:g}_tau{args.tau_w:g}_f{fraction:g}_off{off:g}_bias{bias:g}'
   if (D/(label+'.json')).exists():continue
   c.configure(fraction,off,bias);x,d=c.run();d['label']=label
   np.savez_compressed(D/(label+'.npz'),free=x[0],bound=x[1]);(D/(label+'.json')).write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d),flush=True)
