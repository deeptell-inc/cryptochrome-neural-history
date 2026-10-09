from pathlib import Path
import sys,json,time,argparse,gc
from types import SimpleNamespace
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl.hq_rotation import frozen,s
from impl.hq_electric import electric_basis
from impl.hq_binding import BindingCycle
from impl.hq_field_couplings import matched_h_response,coupled_generator,BoundFieldModel,well
D=ROOT/'data/field-couplings';kappa=10.910941626120245
p=argparse.ArgumentParser();p.add_argument('--L',type=int,default=2);p.add_argument('--mode',choices=['well','electronic','combined','baseline'],required=True);p.add_argument('--angles',nargs='+',type=float,default=[1.,5.]);args=p.parse_args()
with threadpool_limits(limits=1):
 f=frozen();response=({st:np.zeros((3,len(h),len(h)),complex) for st,h in f['body'].items()} if args.mode in ('well','baseline') else matched_h_response()[0])
 field=14. if args.mode in ('electronic','combined') else 0.;deform=args.mode in ('well','combined')
 angular=electric_basis(args.L,kappa);generator=coupled_generator(response,field);reference=None
 for angle in args.angles:
  label=f'{args.mode}_L{args.L}_angle{angle:g}';t=time.monotonic()
  model=BoundFieldModel(f,response,field,kappa,angle,deform=deform)
  c=BindingCycle(f,args.L,rms_deg=angle,angular_override=angular,generator_override=generator,node_model=model.at,mobile_reference=reference)
  if reference is None:reference=SimpleNamespace(cutoff=c.cutoff,tau_free=c.tau_free,field=c.field,gf=c.gf,afrp=c.afrp)
  # Population boundary must use the new free HQ Hamiltonian at every orientation.
  projs=[]
  for n in angular['n']:
   h=f['body']['H']+field*np.einsum('a,aij->ij',n,response['H'])+c.field*sum(n[a]*f['Z']['H'][a] for a in range(3))
   _,u=eigh(h);cols=np.column_stack([s.vec(np.outer(u[:,j],u[:,j].conj())) for j in range(9)]);projs.append(np.einsum('ij,kj->ik',cols,cols.conj()))
  c.projectors['f']=np.array(projs)
  if args.mode=='electronic' and angle==args.angles[0]:
   c.configure(0.,0.,0.);xx,rr=c.run();rr.update(kappa=kappa,electronic_field_MVm=field)
   (D/f'free_electronic_L{args.L}.json').write_text(json.dumps(rr,indent=2)+'\n')
  raw=np.array([np.exp(-well(n,angle,1e-11,kappa if deform else 0.)['free_energy_residual']) for n in angular['n']]);Z=angular['weights']@raw
  c.configure(.5,1.,0.,profile_override=raw/Z);x,r=c.run()
  r.pop('validity',None);r['field_wobble_validity']=model.validity
  r.update(label=label,electronic_field_MVm=field,kappa=kappa,well_deformation=deform,laplace_partition_normalization=float(Z),total_seconds=time.monotonic()-t,mode=args.mode,local_intrinsic_bath='frozen prior bath; no new electric fluctuation spectrum',native_calibrated=False)
  (D/(label+'.json')).write_text(json.dumps(r,indent=2)+'\n');np.savez_compressed(D/(label+'.npz'),free=x[0],bound=x[1]);(D/(label+'_well.json')).write_text(json.dumps(list(model.records.values()),indent=2)+'\n');print(json.dumps(r),flush=True)
  del c,model;gc.collect()
