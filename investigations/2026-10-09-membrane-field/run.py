from pathlib import Path
import sys,json,time,argparse,gc
from types import SimpleNamespace
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl.hq_rotation import frozen
from impl.hq_electric import ElectricCycle,electric_generator
from impl.hq_binding import BindingCycle
p=argparse.ArgumentParser();p.add_argument('--L',type=int,default=2);p.add_argument('--kappas',nargs='+',type=float,default=[0.,.10910941626120247,1.0910941626120247,10.910941626120245]);p.add_argument('--axes',nargs='+',type=int,default=[2]);p.add_argument('--bound',action='store_true');args=p.parse_args()
D=ROOT/'data/membrane-field'
def save(label,x,r):
 r['label']=label;(D/(label+'.json')).write_text(json.dumps(r,indent=2)+'\n');np.savez_compressed(D/(label+'.npz'),**x);print(json.dumps(dict(label=label,delta=r['delta_yield'],yields=r['yields'],seconds=r.get('seconds'))),flush=True)
with threadpool_limits(limits=1):
 f=frozen()
 for axis in args.axes:
  for k in args.kappas:
   t=time.monotonic();c=ElectricCycle(f,args.L,k,axis);x,r=c.run();r.update(kappa=k,cutoff=args.L,axis=axis,seconds=time.monotonic()-t)
   save(f'free_L{args.L}_k{k:.8g}_axis{axis}',dict(density=x),r)
   if args.bound:
    ref=SimpleNamespace(cutoff=args.L,tau_free=c.tau_s,field=c.field,gf=c.generators,afrp=c.solves['RP'])
    for angle in [1.,5.]:
     b=BindingCycle(f,args.L,rms_deg=angle,angular_override=c.angular,generator_override=electric_generator,mobile_reference=ref)
     for off in ([1.,100.] if angle==1. and k in (0.,10.910941626120245) else [1.]):
      b.configure(.5,off,0.);x,r=b.run();r.update(kappa=k,axis=axis)
      save(f'bound_L{args.L}_k{k:.8g}_axis{axis}_angle{angle:g}_off{off:g}',dict(free=x[0],bound=x[1]),r)
     del b;gc.collect()
   del c;gc.collect()
