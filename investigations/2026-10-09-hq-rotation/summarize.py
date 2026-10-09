from pathlib import Path
import json,hashlib,sys,platform,csv
import numpy as np,scipy
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'data/hq-rotation'
read=lambda name:json.loads((D/name).read_text())
validation=read('validation.json');assert validation['status']=='passed'
rows=[]
for tag in ['9.72','21.476596','99.12']:
 a=read(f'L3_tau{tag}_B50.json');b=read(f'independent_L4_tau{tag}.json')
 assert a['trace_error']<1e-10 and a['hermiticity_error']<1e-10
 assert a['minimum_angular_density_eigenvalue']>-1e-10
 assert max(a['solve_residuals'].values())<1e-10
 assert max(a['yields'].values())-min(a['yields'].values())<1e-10
 assert abs(a['yields']['reset']-b['yield_reset_uniform'])<2e-6
 rows.append(dict(tau_ns=a['tau_ns'],full=a['yields']['full'],population=a['yields']['population'],reset=a['yields']['reset'],independent_L4=b['yield_reset_uniform'],L3_L4_difference=abs(a['yields']['reset']-b['yield_reset_uniform']),trace_error=a['trace_error'],direct_spin_contrast_diagnostic=a['direct_spin_contrast']))
with (D/'summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
a=read('L3_tau21.476596_B50.json');b=read('independent_L3_tau21.476596.json')
assert abs(a['yields']['reset']-b['yield_reset_uniform'])<1e-10
assert abs(read('independent_L5_tau99.12.json')['yield_reset_uniform']-read('independent_L6_tau99.12.json')['yield_reset_uniform'])<1e-8
spectra=read('HQ_spectrum.json');assert all(x['independent_relative_error']<1e-8 for x in spectra)
sources=[ROOT/'HQ_ROTATION_RESULTS.md',ROOT/'data/population-memory/latest_case.npz',ROOT/'data/population-memory/modes_and_states.npz',ROOT/'impl/hq_rotation.py',ROOT/'impl/dark_threshold_source.py',ROOT/'impl/dark_md_coefficients.py']+list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.sh'))+list(Path(__file__).parent.glob('*.md'))
hashfile=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['latest_case.npz','modes_and_states.npz']:
 assert hashfile(ROOT/'data/population-memory'/name)==hashfile(ROOT/'paper/journal-of-biological-physics/supplementary-data/data/population-memory'/name)
manifest=dict(date='2026-10-09',validation='passed',python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),sources={str(p.relative_to(ROOT)):hashfile(p) for p in sources},outputs={str(p.relative_to(ROOT)):hashfile(p) for p in D.iterdir() if p.suffix in ['.json','.npz','.csv'] and p.name!='manifest.json'},historical_inputs_identical_to_submission_archive=True,report_resolution_for_contrast=1e-10,readout_discretization_check='L3 versus independent L4; slow rotation additionally L5 and L6',physical_scope='conditional free isotropic common rigid-body rotation; uncalibrated in vivo')
(D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(status='passed',rows=rows),indent=2))
