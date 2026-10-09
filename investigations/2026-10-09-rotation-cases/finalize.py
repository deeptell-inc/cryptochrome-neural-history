from pathlib import Path
import json,hashlib,ast,platform,sys
import numpy as np,scipy
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'data/rotation-cases';HERE=Path(__file__).resolve().parent
hashfile=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prior=json.loads((ROOT/'data/hq-rotation/manifest.json').read_text())
for name,digest in prior['sources'].items():assert hashfile(ROOT/name)==digest,name
checks=json.loads((D/'checks.json').read_text());assert checks['status']=='passed' and checks['archived_reference_trial_files_exact']==18
storage=json.loads((D/'protected_wait_storage.json').read_text());assert storage['independent_population_max_error']<1e-9
assert round(storage['half_decay_s'],5)==round(storage['independent_population_half_s'],5)
files=[ROOT/'impl/hq_rotation_cases.py',*HERE.glob('*.py'),*HERE.glob('*.sh'),*HERE.glob('*.md'),ROOT/'ROTATION_CASES_RESULTS.md']
for p in files:
 if p.suffix=='.py':ast.parse(p.read_text())
for name in ['synaptic_veto','sparse_synaptic','stochastic_branch_spike']:
 assert (ROOT/f'impl/{name}.py').read_bytes()==(ROOT/f'paper/journal-of-biological-physics/supplementary-data/impl/{name}.py').read_bytes()
plans=[json.loads(p.read_text()) for p in (D/'reference-runtime').glob('neural_plan_*.json')]
primary=list(D.glob('molecular*.json'))+list(D.glob('molecular*.npz'))+list(D.glob('*summary.csv'))+[D/'neural_selected.csv',D/'checks.json',D/'protected_wait_storage.json',D/'independent_chemical_storage.json',D/'case_summary.png',D/'case_summary.pdf']+list((D/'reference-runtime').glob('*'))
inputs=['data/population-memory/latest_case.npz','data/population-memory/modes_and_states.npz','data/dark-basis-resolution/conditional_products_spikes.csv','data/hq-rotation/manifest.json','impl/hq_rotation.py','impl/synaptic_veto.py','impl/sparse_synaptic.py','impl/stochastic_branch_spike.py','impl/literature_bridge_recalculation.py','impl/chemical_carrier_mapping.py']
manifest=dict(date='2026-10-09',status='passed',previous_rotation_sources_unchanged=True,archived_neural_code_unchanged=True,neural_primary_runtime=json.loads((D/'reference-runtime/runtime.json').read_text()),molecular_runtime=prior['python'],molecular_numpy=prior['numpy'],unique_neural_condition_specs=sum(map(len,plans)),paired_trials=sum(p['n']*len(p['seeds']) for plan in plans for p in plan),source_hashes={str(p.relative_to(ROOT)):hashfile(p) for p in files},input_hashes={n:hashfile(ROOT/n) for n in inputs},primary_output_hashes={str(p.relative_to(ROOT)):hashfile(p) for p in primary},nonprimary='Root-level neural_mixture/routes/fast outputs made with NumPy 2.2.6 are runtime diagnostics; reference-runtime/ NumPy 2.5.0 outputs exclusively feed final tables.',biological_calibration=False,publication_performed=False)
(D/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n');print(json.dumps({k:manifest[k] for k in ['status','unique_neural_condition_specs','paired_trials','previous_rotation_sources_unchanged','archived_neural_code_unchanged']},indent=2))
