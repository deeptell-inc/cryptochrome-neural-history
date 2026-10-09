from pathlib import Path
import sys,json,argparse,time
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl.hq_rotation import frozen
from impl.hq_binding import BindingCycle
p=argparse.ArgumentParser();p.add_argument('--angle',type=float,default=0.);p.add_argument('--tau-w',type=float,default=1e-11);p.add_argument('--cutoff',type=int,default=2);p.add_argument('--pilot',action='store_true');p.add_argument('--verify',action='store_true');args=p.parse_args()
D=ROOT/'data/binding-motion';D.mkdir(exist_ok=True)
with threadpool_limits(limits=1):
 t=time.monotonic();c=BindingCycle(frozen(),args.cutoff,rms_deg=args.angle,tau_w=args.tau_w);print(json.dumps(dict(built_seconds=time.monotonic()-t)),flush=True)
 specs=[(.5,1.,0.)] if args.pilot else [(f,k,0.) for f in (.01,.1,.5,.9) for k in (.01,1.,100.)]
 if not args.pilot:specs += [(.5,1.,a) for a in (-.9,1.8)]+[(0.,0.,0.),(.5,0.,0.)]
 if args.verify:specs=[(.5,.01,0.),(.5,1.,0.),(.5,100.,0.),(.5,1.,1.8),(.5,0.,0.)]
 for f,k,a in specs:
  label=f'L{args.cutoff}_angle{args.angle:g}_tau{args.tau_w:g}_f{f:g}_off{k:g}_bias{a:g}'
  if (D/(label+'.json')).exists():continue
  c.configure(f,k,a);x,result=c.run();result['label']=label
  np.savez_compressed(D/(label+'.npz'),free=x[0],bound=x[1]);(D/(label+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
