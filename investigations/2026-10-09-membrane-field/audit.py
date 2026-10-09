from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'data/membrane-field'
c=json.loads((D/'checks.json').read_text());m=json.loads((D/'manifest.json').read_text())
assert m['case_count']==18
for group in ('source_files','result_files'):
 for path,h in m[group].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
assert m['source_files']['data/population-memory/latest_case.npz']=='5e139d067e38f095390a55e05c5b8b982f5f24734514ee198eb00ced74f285c2'
assert len(c['cutoff_comparisons'])==4
assert max(x['max_yield_difference'] for x in c['cutoff_comparisons'])<1e-5
assert max(abs(x['delta_difference']) for x in c['cutoff_comparisons'])<1e-6
assert c['diagnostics']['max_trace']<1e-10
assert c['diagnostics']['max_hermiticity']<1e-9
assert c['diagnostics']['min_sampled_eigenvalue']>=-1e-10
assert c['diagnostics']['max_full_population_difference']<1e-10
sp=json.loads((D/'HQ_spectrum.json').read_text());dif=[]
for x in sp:
 if x['cutoff']==4:
  y=next(y for y in sp if (y['axis'],y['kappa'],y['cutoff'])==(x['axis'],x['kappa'],3));dif.append(abs(x['inverse_slowest_us']/y['inverse_slowest_us']-1))
assert max(dif)<.01
assert max(x.get('independent_relative_error',0) for x in sp)<1e-8
assert len(json.loads((D/'axial_limit.json').read_text()))==3
for p in D.glob('*_L*_k*_axis*.json'):
 r=json.loads(p.read_text());assert all(0<=v<=1 for v in r['yields'].values()),p
 if p.name.startswith('free'):assert abs(r['delta_yield'])<1e-10
report=dict(passed=True,finite_cycle_runs=18,unique_finite_cases=14,ideal_axis_cases=3,HQ_spectrum_runs=len(sp),max_HQ_L3_L4_relative_difference=max(dif),checks='hashes, frozen input, cutoff tolerances, density diagnostics, free contrast, independent spectrum')
(D/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
