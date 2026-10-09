from pathlib import Path
import sys,json,csv,hashlib,os,tempfile,platform
import numpy as np
import scipy
from scipy.constants import physical_constants,k
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'data/field-couplings';records=[];source=[];checks={}
for st in ('H','R','E'):
 a=json.loads((D/st/'response.json').read_text());b=json.loads((D/st/'finite_checks.json').read_text());source.append(a);checks[st]=b
 for n in a['nuclei']:records.append(dict(state=st,**n))
with (D/'electronic_summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
rows=[]
for p in sorted(D.glob('*_L[23]_angle[15].json')):
 r=json.loads(p.read_text());rows.append(r)
fields=['mode','cutoff','rms_deg','delta_yield','full_population_difference','trace_error','hermiticity_error','minimum_density_eigenvalue']
with (D/'cycle_summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields+['q_full','q_reset'],extrasaction='ignore');w.writeheader()
 for r in rows:w.writerow({**r,'q_full':r['yields']['full'],'q_reset':r['yields']['reset']})
compare=[];baseline=[]
for r in rows:
 if r['cutoff']==3:
  old=next(x for x in rows if x['mode']==r['mode'] and x['rms_deg']==r['rms_deg'] and x['cutoff']==2)
  compare.append(dict(angle=r['rms_deg'],delta_difference=r['delta_yield']-old['delta_yield'],max_yield_difference=max(abs(r['yields'][key]-old['yields'][key]) for key in r['yields'])))
 if r['mode']=='baseline':
  old=json.loads((ROOT/'data/membrane-field'/f"bound_L2_k10.910942_axis2_angle{r['rms_deg']:g}_off1.json").read_text())
  baseline.append(dict(angle=r['rms_deg'],delta_difference=r['delta_yield']-old['delta_yield'],max_yield_difference=max(abs(r['yields'][key]-old['yields'][key]) for key in r['yields'])))
checks.update(cutoff_comparisons=compare,baseline_regression=baseline)
projection=[]
for r in rows:
 p=D/'before_hermitian_projection'/f"{r['mode']}_L{r['cutoff']}_angle{r['rms_deg']:g}.json"
 if p.exists():
  old=json.loads(p.read_text());projection.append(dict(mode=r['mode'],angle=r['rms_deg'],max_yield_change=max(abs(r['yields'][key]-old['yields'][key]) for key in r['yields']),delta_change=r['delta_yield']-old['delta_yield']))
checks['hermitian_projection_regression']=projection
checks['cycle_diagnostics']=dict(max_trace=max(r['trace_error'] for r in rows),max_hermiticity=max(r['hermiticity_error'] for r in rows),min_eigenvalue=min(r['minimum_density_eigenvalue'] for r in rows),max_full_population=max(abs(r['full_population_difference']) for r in rows))
checks['independent_cycle']=json.loads((D/'independent_cycle.json').read_text())
wg=json.loads((D/'well_independent.json').read_text());checks['well_quadrature']=dict(max_harmonic_covariance_relative_error=max(x['covariance_relative_error'] for x in wg),max_7_11_quadrature_difference=max(x.get('quadrature_covariance_relative_change',0) for x in wg))
# State-specific fragment polarizability energy scales; not the full protein polarizability.
polar=[]
for x in source:
 alpha=np.array(x['checks']['polarizability_au']);ev=np.linalg.eigvalsh((alpha+alpha.T)/2)
 factor=.5*(14e6/physical_constants['atomic unit of electric field'][0])**2*physical_constants['Hartree energy'][0]/(k*310)
 polar.append(dict(state=x['state'],max_induced_energy_kBT=float(factor*ev.max()),orientation_energy_span_kBT=float(factor*np.ptp(ev))))
checks['fragment_polarizability']=polar
(D/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'qvolition-field-mpl'));os.environ.setdefault('XDG_CACHE_HOME',str(Path(tempfile.gettempdir())/'qvolition-field-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(11,3.8),layout='constrained');plt.rcParams.update({'font.size':10})
labels=[f"{r['state']}-{r['atom']}" for r in records];ax[0].bar(labels,[100*r['EFG_max_relative_change_at_14MVm'] for r in records]);ax[0].set(ylabel='Maximum EFG tensor change (%)',title='A  Matched electronic response at 14 MV/m');ax[0].tick_params(axis='x',rotation=35)
mods=['baseline','electronic','well','combined'];names=['Orientation only','+ Electronic','+ Well','+ Both']
for angle in (1.,5.):
 base=next(r['delta_yield'] for r in rows if r['mode']=='baseline' and r['rms_deg']==angle)
 vals=[next(r['delta_yield'] for r in rows if r['mode']==m and r['rms_deg']==angle and r['cutoff']==2)/base for m in mods]
 ax[1].plot(names,vals,'o-',label=f'{angle:g} deg bound wobble')
ax[1].axhline(1,color='grey',ls=':',lw=1);ax[1].set(ylabel='History contrast / orientation-only contrast',title='B  Separate electronic and well effects');ax[1].legend(frameon=False);ax[1].tick_params(axis='x',rotation=20)
for a in ax:a.spines[['top','right']].set_visible(False)
fig.savefig(D/'field_coupling_cases.png',dpi=180);fig.savefig(D/'field_coupling_cases.pdf');plt.close(fig)
files=[ROOT/'impl/hq_field_couplings.py',ROOT/'impl/hq_binding.py',ROOT/'impl/hq_electric.py',ROOT/'impl/hq_rotation.py',ROOT/'tests/test_hq_field_couplings.py',ROOT/'data/population-memory/latest_case.npz']+list((ROOT/'investigations/2026-10-09-field-couplings').glob('*'))
files += [ROOT/'impl'/name for name in ('dark_electronic_response.py','dark_electronic_tensors.py','dark_embedded_tensors.py','dark_md_coefficients.py','dark_threshold_source.py')]
files += [ROOT/'data/dark-md-coefficients'/folder/name for folder in ('seed_2026091521','seed_2026091521_dense') for name in ('trajectory.npz','metadata.json')]
files += [ROOT/'data/dcry-mobile-model/topology.json',D/'before_hermitian_projection/hq_field_couplings.py']
files += [ROOT/'data/dark-electronic-tensors'/state/'input.json' for state in ('H_anionic_HQ','R_oxidized','E_neutral_SQ')]
for entry in source:
 p=ROOT/entry['source']/'density.npz'
 if p.exists():files.append(p)
manifest=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),source_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()},electronic_input_hashes={a['state']:a['source_hashes'] for a in source},cycle_run_count=len(rows),scope='conditional fixed-geometry static response; conditional harmonic binding; frozen intrinsic local bath; no native prediction')
manifest['result_files']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in D.rglob('*') if p.is_file() and p.suffix in ('.json','.npz','.csv','.png','.pdf') and p.name not in ('manifest.json','acceptance.json')}
(D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(checks,indent=2))
