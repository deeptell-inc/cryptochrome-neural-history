from pathlib import Path
import json,hashlib,csv
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'data/field-couplings'
read=lambda p:json.loads((D/p).read_text())
c=read('checks.json');m=read('manifest.json')
assert m['cycle_run_count']==10
for group in ('source_files','result_files'):
 for path,h in m[group].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
for group in m['electronic_input_hashes'].values():
 for path,h in group.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
assert m['source_files']['data/population-memory/latest_case.npz']=='5e139d067e38f095390a55e05c5b8b982f5f24734514ee198eb00ced74f285c2'
for state in ('H','R','E'):
 r=read(f'{state}/response.json')
 assert r['baseline_EFG_relative_error']<1e-7
 assert r['baseline_HFC_relative_error'] is None or r['baseline_HFC_relative_error']<1e-7
 assert r['checks']['CPHF_relative_residual']<1e-7
 assert r['checks']['electron_number_derivative_max']<1e-7
 for q in c[state]['checks']:
  assert q['EFG_derivative_relative_error']<.01
  assert q.get('HFC_derivative_relative_error',0)<.01
assert c['E']['baseline_gradient']<1e-6
assert len(c['cutoff_comparisons'])==2
assert max(x['max_yield_difference'] for x in c['cutoff_comparisons'])<1e-5
assert max(abs(x['delta_difference']) for x in c['cutoff_comparisons'])<1e-6
assert max(x['max_yield_difference'] for x in c['baseline_regression'])<1e-9
assert len(c['hermitian_projection_regression'])==8
assert max(x['max_yield_change'] for x in c['hermitian_projection_regression'])<1e-8
diag=c['cycle_diagnostics'];assert diag['max_trace']<1e-10 and diag['max_hermiticity']<1e-9
assert diag['min_eigenvalue']>=-1e-9
assert c['independent_cycle']['direct_H_state_error']<2e-8
assert c['independent_cycle']['direct_H_residual']<1e-8
assert c['independent_cycle']['third_rank_transform_error']<1e-9
assert c['well_quadrature']['max_harmonic_covariance_relative_error']<.001
assert c['well_quadrature']['max_7_11_quadrature_difference']<1e-7
sp=read('HQ_spectrum.json');diff=[]
assert len(sp)==9
for row in sp:
 if row['cutoff']==4:
  old=next(x for x in sp if x['field_MVm']==row['field_MVm'] and x['cutoff']==3)
  diff.append(abs(row['inverse_slowest_us']/old['inverse_slowest_us']-1))
assert max(diff)<.01
assert max(x.get('independent_relative_error',0) for x in sp)<1e-8
assert len(list(csv.DictReader((D/'well_cases.csv').open())))==50
for row in read('well_averages.json'):
 assert row['exact_global_bound_fraction_isotropic_anchors']==.5
for p in D.glob('*_L[23]_angle[15].json'):
 r=read(p.name);assert all(0<=x<=1 for x in r['yields'].values())
 for v in r.get('field_wobble_validity',{}).values():
  assert v['max_bandwidth_tau']<.01
  assert v['max_unital_defect']<1e-9
free=read('free_electronic_L2.json');assert abs(free['delta_yield'])<1e-10
assert '14 passed' in (D/'tests.log').read_text()
report=(ROOT/'FIELD_COUPLINGS_RESULTS.md').read_text();assert '\t' not in report and '\r' not in report
out=dict(passed=True,cycle_runs=10,free_runs=1,well_conditions=50,electronic_states=3,finite_field_pairs=5,HQ_spectrum_runs=9,max_spectrum_L3_L4_relative_difference=max(diff),report_sha256=hashlib.sha256(report.encode()).hexdigest(),checks='input and result hashes; frozen case; finite-field derivatives; independent well/HQ/spectrum; cutoff; density; regression; 14 tests',scope='conditional numerical validation, not native calibration')
(D/'acceptance.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
