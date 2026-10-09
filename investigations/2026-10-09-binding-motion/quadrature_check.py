from pathlib import Path
import sys,json,gc
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl.hq_rotation import frozen
from impl.hq_binding import BindingCycle
D=ROOT/'data/binding-motion';Q=D/'quadrature-check';Q.mkdir(exist_ok=True)
with threadpool_limits(limits=1):
 f=frozen();previous=None;checks=[]
 for angle,biases in [(0.,[0.,1.8]),(1.,[1.8])]:
  c=BindingCycle(f,3,rms_deg=angle,mobile_reference=previous,quadrature_extra=3);previous=c;gc.collect()
  for bias in biases:
   label=f'L3_angle{angle:g}_tau1e-11_f0.5_off1_bias{bias:g}'
   c.configure(.5,1.,bias);x,d=c.run();d['label']=label
   reference=json.loads((D/(label+'.json')).read_text())
   d['delta_difference_from_55_nodes']=abs(d['delta_yield']-reference['delta_yield'])
   d['yield_difference_from_55_nodes']=max(abs(d['yields'][key]-reference['yields'][key]) for key in d['yields'])
   (Q/(label+'.json')).write_text(json.dumps(d,indent=2)+'\n');checks.append(d)
   print(json.dumps(d),flush=True)
 (Q/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
