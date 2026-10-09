from pathlib import Path
import sys,json,time
from dataclasses import replace
from unittest.mock import patch
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from scipy.stats import beta
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl import synaptic_veto as v
D=ROOT/'data/binding-motion';N=D/'neural';N.mkdir(exist_ok=True)
assert np.__version__=='2.5.0','Use archived neural runtime'
(N/'runtime.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__),indent=2)+'\n')
# Three dwell times, four wobble amplitudes, then fraction/orientation controls.
keys=[(angle,.5,off,0.) for angle in (0.,.1,1.,5.) for off in (.01,1.,100.)]
keys += [(1.,f,1.,0.) for f in (.01,.9)]+[(1.,.5,1.,b) for b in (-.9,1.8)]+[(0.,.5,0.,0.),(0.,0.,0.,0.)]
seeds=[26091711,26091712,26091713];n=8192;rows=[];plan=[]
with threadpool_limits(limits=1):
 for index,(angle,fraction,off,bias) in enumerate(keys):
  tail=f'angle{angle:g}_tau1e-11_f{fraction:g}_off{off:g}_bias{bias:g}'
  paths=sorted(D.glob('L*_'+tail+'.json'));assert paths,tail
  d=json.loads(paths[-1].read_text());q0=d['yields']['reset'];q1=d['yields']['full']
  if fraction==0:q1=q0
  p=dict(index=index,label=tail,molecular_file=paths[-1].name,q0=q0,q1=q1,n=n,seeds=seeds,route='stop',gain=1024,pool=10000,tau_s=.2,allocation='top8');plan.append(p)
  blocks=[]
  for seed in seeds:
   out=N/f'{index}_{seed}.npz'
   if out.exists():
    z=np.load(out);assert float(z['q0'])==q0 and float(z['q1'])==q1;oc=z['outcomes'];times=z['times']
   else:
    cfg=replace(v.Config(),gain=1024.,route='stop',tau=.2)
    with patch.object(v,'sources',return_value=(q0,dict(full=q1,population=q1,reset=q0))):r=v.simulate(cfg,n=n,seed=seed,names=('top',))
    oc=r['outcomes'];times=r['times'];np.savez_compressed(out,outcomes=oc,times=times,q0=q0,q1=q1)
    raw=v.summarize(r);assert min(a['p_min'] for a in raw)>=0 and max(a['p_max'] for a in raw)<=1;assert max(a['budget_error'] for a in raw)<1e-10
   go_time=times[...,0];stop=np.isfinite(times[...,2])&(times[...,2]<go_time);go=np.isfinite(go_time)&~stop
   assert np.array_equal(oc,np.stack([stop,go,~(stop|go)],axis=-1))
   if q0==q1:assert np.array_equal(oc[:,:,0],oc[:,:,1])
   blocks.append(oc)
  outcomes=np.concatenate(blocks,axis=3)
  for mi,mode in enumerate(('majority','weighted','temporal','competition')):
   changes=outcomes[mi,0,1,:,0].astype(int)-outcomes[mi,0,0,:,0].astype(int);nn=len(changes);plus=int((changes==1).sum());minus=int((changes==-1).sum())
   def interval(k):return (0. if k==0 else beta.ppf(.0125,k,nn-k+1),1. if k==nn else beta.ppf(.9875,k+1,nn-k))
   pl,pu=interval(plus);ml,mu=interval(minus)
   rows.append(dict(**p,integration=mode,delta_stop_pp=100*changes.mean(),low_pp=100*(pl-mu),high_pp=100*(pu-ml),paired_trials=nn,plus=plus,minus=minus))
  pd.DataFrame(rows).to_csv(N/'summary.csv',index=False);print(json.dumps(dict(index=index,label=tail,temporal=rows[-2]['delta_stop_pp'])),flush=True)
 (N/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
