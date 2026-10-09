from pathlib import Path
import json,sys,hashlib,platform
import numpy as np,pandas as pd
from scipy.stats import beta
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'data/rotation-cases';N=D/'reference-runtime'
read=lambda p:json.loads(p.read_text())
rows=[];checks={}
for batch in ['mixture','routes','fast','field']:
 plan=read(N/f'neural_plan_{batch}.json')
 for index,p in enumerate(plan):
  blocks=[]
  for seed in p['seeds']:
   z=np.load(N/f'neural_{batch}_{index}_{seed}.npz');o=z['outcomes'];times=z['times']
   g=times[...,0];stop=np.isfinite(times[...,2])&(times[...,2]<g);go=np.isfinite(g)&~stop
   assert np.array_equal(o,np.stack([stop,go,~(stop|go)],axis=-1));assert np.all(o.sum(-1)==1)
   if p['q0']==p['q1']:assert np.array_equal(o[:,:,0],o[:,:,1])
   blocks.append(o)
  outcomes=np.concatenate(blocks,axis=3)
  for mi,mode in enumerate(['majority','weighted','temporal','competition']):
   for ai,alloc in enumerate(p['names']):
    d=outcomes[mi,ai,1,:,0].astype(int)-outcomes[mi,ai,0,:,0].astype(int);n=len(d)
    delta=100*d.mean();se=100*d.std(ddof=1)/np.sqrt(n)
    plus=int((d==1).sum());minus=int((d==-1).sum())
    def interval(k):
     return (0. if k==0 else beta.ppf(.0125,k,n-k+1),1. if k==n else beta.ppf(.9875,k+1,n-k))
    pl,pu=interval(plus);ml,mu=interval(minus)
    diff=abs(p['q1']-p['q0']);bound=100*(-np.expm1(10000*np.log1p(-diff)))
    rows.append(dict(batch=batch,scenario=p['label'],route=p['route'],tau_s=p['tau_s'],allocation=alloc,integration=mode,n=n,q0=p['q0'],q1=p['q1'],delta_q=p['q1']-p['q0'],delta_stop_pp=delta,MC_se_pp=se,normal_low_pp=delta-1.96*se,normal_high_pp=delta+1.96*se,conservative_low_pp=100*(pl-mu),conservative_high_pp=100*(pu-ml),disagreements=plus+minus,ref_stop_probability=float(outcomes[mi,ai,0,:,0].mean()),source_stop_probability=float(outcomes[mi,ai,1,:,0].mean()),one_cue_coupling_bound_pp=bound))
 pdrows=pd.read_csv(N/f'neural_{batch}_all.csv');assert pdrows.p_min.min()>=0 and pdrows.p_max.max()<=1;assert pdrows.budget_error.max()<1e-10
checks['stored_events_reclassified']=True;checks['zero_input_arms_identical']=True
old=pd.read_csv(ROOT/'data/synaptic-veto/seed_summary.csv');plan=read(N/'neural_plan_routes.json');compared=0
for index,p in enumerate(plan):
 if p['label']!='clamped':continue
 for seed in p['seeds']:
  match=old[(old['group']=='confirmation')&(old.route==p['route'])&(old.tau==p['tau_s'])&(old.seed==seed)]
  key=match.iloc[0]['condition_id'];a=np.load(ROOT/f'data/synaptic-veto/trials/{key}_{seed}.npz');b=np.load(N/f'neural_routes_{index}_{seed}.npz')
  for what in ['times','outcomes']:assert np.array_equal(a[what][:,2:3],b[what]),(p,seed,what)
  compared+=1
checks['archived_reference_trial_files_exact']=compared
frame=pd.DataFrame(rows);frame.to_csv(D/'neural_summary.csv',index=False)
small=frame.query('integration=="temporal" and allocation=="top"');small.to_csv(D/'neural_selected.csv',index=False)
mol=[]
for path in sorted(D.glob('molecular_L2_*.json')):
 r=read(path)
 if not isinstance(r,dict):continue
 p3=D/f"molecular_L3_{r['label']}.json";best=read(p3) if p3.exists() else r
 assert best['trace_error']<1e-8 and best['hermiticity_error']<1e-8 and best['minimum_angular_density_eigenvalue']>-1e-9
 error=abs(best['delta_yield']-r['delta_yield']);assert error<1e-7
 mol.append(dict(case=best['label'],cutoff=best['cutoff'],full=best['yields']['full'],population=best['yields']['population'],reset=best['yields']['reset'],delta=best['delta_yield'],full_population_gap=best['full_population_difference'],mean_preparation_time_s=best['mean_preparation_time_s'],L2_L3_delta_gap=error if p3.exists() else None))
pd.DataFrame(mol).to_csv(D/'molecular_summary.csv',index=False)
qerr=1e-10;checks['unresolved_delta_sensitivity_bound_pp']=float(100*(-np.expm1(10000*np.log1p(-qerr))))
checks['independent_molecular']=read(D/'molecular_independent_checks.json');assert checks['independent_molecular']['status']=='passed'
chemical=read(D/'independent_chemical_storage.json');assert max(r['independent_ode_error'] for r in chemical)<1e-8
routing=[]
for row in mol:
 for t in [0.,.012,.2,1.]:routing.append(dict(case=row['case'],time_s=t,delta_peroxide_equivalents=.5*row['delta']*np.exp(-4*t),assumed_complete_convergent_chemistry=True))
pd.DataFrame(routing).to_csv(D/'routing_summary.csv',index=False)
checks['status']='passed';(D/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(small[['scenario','route','tau_s','delta_stop_pp','normal_low_pp','normal_high_pp','conservative_low_pp','conservative_high_pp']].to_string(index=False))
