from pathlib import Path
import sys,json,csv,hashlib,platform,os,tempfile
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl.hq_electric import electric_basis
from impl.hq_rotation import s
D=ROOT/'data/membrane-field';rows=[]
for p in sorted(D.glob('*_L*_k*_axis*.json')):
 r=json.loads(p.read_text());r['kind']='bound' if p.name.startswith('bound') else 'free';r['file']=p.name
 rows.append(r)
fields=['file','kind','cutoff','kappa','axis','rms_deg','koff_s','delta_yield','direct_spin_contrast','full_population_difference','trace_error','hermiticity_error','minimum_density_eigenvalue']
with (D/'summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields+['q_full','q_population','q_reset'],extrasaction='ignore');w.writeheader()
 for r in rows:w.writerow({**r,**{'q_'+k:v for k,v in r['yields'].items()}})
# Direct moments of nuclear spin in laboratory B direction. Sum over two 14N nuclei.
spinrows=[]
for r in rows:
 p=D/r['file'];npz=p.with_suffix('.npz')
 if not npz.exists():continue
 a=electric_basis(r['cutoff'],r['kappa'],r['axis']);z=dict(np.load(npz));state=np.einsum('qa,aik->qik',a['Y'],z.get('density',z.get('free')))
 if 'bound' in z:state+=z['bound']
 polar=[];align=[]
 for n in a['n']:
  ops=[sum(n[j]*s.NI[i][j] for j in range(3)) for i in range(2)]
  polar.append(sum(ops));align.append(sum(3*np.einsum('ij,jk->ik',v,v)-2*np.eye(9) for v in ops))
 # trace(O rho)=vec(O.T).T vec(rho); vec conventions are column-major.
 for name,ops in [('polarization',polar),('rank2_alignment',align)]:
  bra=np.array([s.vec(o.T) for o in ops]);val=np.einsum('q,qi,qik->k',a['weights'],bra,state)
  spinrows.append(dict(file=r['file'],observable=name,full=float(val[0].real),population=float(val[1].real),reset=float(val[2].real),max_imag=float(max(abs(val.imag)))))
(D/'spin_moments.json').write_text(json.dumps(spinrows,indent=2)+'\n')
# Convergence, diagnostics, and regression against pre-existing results.
check=json.loads((D/'independent_checks.json').read_text());comparisons=[]
for r in rows:
 if r['cutoff']!=3:continue
 p=D/r['file'].replace('_L3_','_L2_')
 if not p.exists():continue
 old=json.loads(p.read_text());comparisons.append(dict(file=r['file'],delta_difference=r['delta_yield']-old['delta_yield'],max_yield_difference=max(abs(r['yields'][m]-old['yields'][m]) for m in r['yields'])))
check['cutoff_comparisons']=comparisons
reg=[]
for r in rows:
 if r['kind']=='bound' and r['kappa']==0 and r['cutoff']==2:
  p=ROOT/'data/binding-motion'/f"L2_angle{r['rms_deg']:g}_tau1e-11_f0.5_off{r['koff_s']:g}_bias0.json"
  old=json.loads(p.read_text());reg.append(dict(file=r['file'],delta_difference=r['delta_yield']-old['delta_yield'],max_yield_difference=max(abs(r['yields'][m]-old['yields'][m]) for m in r['yields'])))
check['zero_field_regression']=reg
check['diagnostics']=dict(max_trace=max(r['trace_error'] for r in rows),max_hermiticity=max(r['hermiticity_error'] for r in rows),min_sampled_eigenvalue=min(r.get('minimum_density_eigenvalue',r.get('minimum_angular_density_eigenvalue')) for r in rows),max_full_population_difference=max(abs(r['full_population_difference']) for r in rows))
(D/'checks.json').write_text(json.dumps(check,indent=2)+'\n')
# Standalone scientific figure.
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"qvolition-membrane-mpl"))
os.environ.setdefault("XDG_CACHE_HOME",str(Path(tempfile.gettempdir())/"qvolition-membrane-cache"))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
kk=np.linspace(0,11,200);lc=np.zeros_like(kk);lc[1:]=1/np.tanh(kk[1:])-1/kk[1:];p2=np.zeros_like(kk);p2[1:]=1-3*lc[1:]/kk[1:]
ax[0].plot(kk,lc,label='Mean cos(theta)');ax[0].plot(kk,p2,label='Molecular P2');ax[0].set(xlabel='kappa = p E / kBT',ylabel='Molecular orientation',title='A  Orientation is biased');ax[0].legend(frameon=False)
spc=json.loads((D/'HQ_spectrum.json').read_text())
for axis in (0,1,2):
 q=sorted([r for r in spc if r['axis']==axis and r['cutoff']==4],key=lambda r:r['kappa']);ax[1].plot([r['kappa'] for r in q],[r['inverse_slowest_us'] for r in q],'o-',label=f'Body dipole axis {"xyz"[axis]}')
ax[1].set(xlabel='kappa = p E / kBT',ylabel='Inverse slowest HQ decay rate (us)',title='B  Free rotation still erases history');ax[1].legend(frameon=False)
for ang in (1,5):
 q=sorted([r for r in rows if r['kind']=='bound' and r['cutoff']==2 and r['axis']==2 and r['rms_deg']==ang and r['koff_s']==1],key=lambda r:r['kappa']);ax[2].semilogy([r['kappa'] for r in q],[abs(r['delta_yield']) for r in q],'o-',label=f'Bound wobble {ang} deg')
ax[2].set(xlabel='kappa = p E / kBT',ylabel='Yield contrast |q_full - q_reset|',title='C  Bound population retains history');ax[2].legend(frameon=False)
fig.savefig(D/'membrane_field_cases.png',dpi=180);fig.savefig(D/'membrane_field_cases.pdf');plt.close(fig)
files=[ROOT/'impl/hq_electric.py',ROOT/'impl/hq_binding.py',ROOT/'impl/hq_rotation.py',ROOT/'tests/test_hq_electric.py',ROOT/'data/population-memory/latest_case.npz']+sorted((ROOT/'investigations/2026-10-09-membrane-field').glob('*'))
manifest=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,data_origin='deterministic conditional simulation; no experimental dataset',source_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()},case_count=len(rows),notes='Local results only; no native dipole/localization calibration, neural recalculation, manuscript revision or remote publication.')
manifest['result_files']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(D.iterdir()) if p.is_file() and p.suffix in ('.json','.npz','.csv','.png','.pdf') and p.name not in ('manifest.json','acceptance.json')}
(D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(check,indent=2));print('Cases',len(rows))
