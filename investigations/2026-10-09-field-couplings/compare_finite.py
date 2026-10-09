"""Compare saved converged finite-field properties with analytic response."""
from pathlib import Path
import json
import numpy as np
D=Path(__file__).resolve().parents[2]/'data/field-couplings'
for state in ('H','R','E'):
 p=D/state;meta=json.loads((p/'response.json').read_text());z=np.load(p/'response.npz');rows=[]
 for amp in ([14e6,7e6] if state in ('H','E') else [14e6]):
  plus=np.load(p/f'finite_{amp:g}_1.npz');minus=np.load(p/f'finite_{amp:g}_-1.npz');n=plus['field_V_m']/amp
  np.testing.assert_allclose(minus['field_V_m'],-plus['field_V_m'],rtol=1e-14)
  dv=np.einsum('a,naij->nij',n,z['dV_au_per_field_au'])*amp/meta['field_au_V_m']
  row=dict(amplitude_V_m=amp,direction=n.tolist(),EFG_derivative_relative_error=float(np.linalg.norm((plus['V_au']-minus['V_au'])/2-dv)/np.linalg.norm(dv)),EFG_even_nonlinearity_relative_to_linear=float(np.linalg.norm((plus['V_au']+minus['V_au'])/2-z['V0_au'])/np.linalg.norm(dv)))
  if state=='E':
   da=np.einsum('a,naij->nij',n,z['dA_MHz_per_field_au'])*amp/meta['field_au_V_m']
   row.update(HFC_derivative_relative_error=float(np.linalg.norm((plus['A_MHz']-minus['A_MHz'])/2-da)/np.linalg.norm(da)),HFC_even_nonlinearity_relative_to_linear=float(np.linalg.norm((plus['A_MHz']+minus['A_MHz'])/2-z['A0_MHz'])/np.linalg.norm(da)))
  rows.append(row)
 prior=p/('finite_acquisition.json' if (p/'finite_acquisition.json').exists() else 'finite_checks.json')
 result={**json.loads(prior.read_text()),'checks':rows,'comparison':'cached converged SCF properties; recomputed independently of the SCF loop'}
 if len(rows)==2:
  estimates=[]
  for amp in (14e6,7e6):
   plus=np.load(p/f'finite_{amp:g}_1.npz');minus=np.load(p/f'finite_{amp:g}_-1.npz')
   estimates.append({name:(plus[name]-minus[name])/(2*amp) for name in ('V_au','A_MHz')})
  result['finite_amplitude_consistency']={name:float(np.linalg.norm(estimates[1][name]-estimates[0][name])/np.linalg.norm(estimates[0][name])) for name in ('V_au','A_MHz') if np.linalg.norm(estimates[0][name])>0}
 (p/'finite_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
