from pathlib import Path
import sys,json,csv
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl.hq_field_couplings import well
from impl.hq_electric import electric_basis
D=ROOT/'data/field-couplings';rows=[]
for ang in (1.,5.):
 for k in (-10.910941626120245,-1.0910941626120247,0.,1.0910941626120247,10.910941626120245):
  for theta in (0.,45.,90.,135.,180.):
   n=np.array([np.sin(np.deg2rad(theta)),0,np.cos(np.deg2rad(theta))]);w=well(n,ang,1e-11,k)
   rows.append(dict(rms0_deg=ang,kappa=k,anchor_angle_deg=theta,mean_shift_deg=w['delta_deg'],rms_deg=w['rms_deg'],tau_min_s=float(min(w['taus'])),tau_max_s=float(max(w['taus'])),F_res_kBT=w['free_energy_residual'],minimum_potential_kBT=w['energy_kBT']))
with (D/'well_cases.csv').open('w') as f:
 wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
averages=[]
for ang in (1.,5.):
 for k in (1.0910941626120247,10.910941626120245):
  a=electric_basis(3,k,nz=10);v=[well(n,ang,1e-11,k) for n in a['n']];ratio=np.array([np.exp(-x['free_energy_residual']) for x in v]);Z=a['weights']@ratio
  averages.append(dict(rms0_deg=ang,kappa=k,partition_ratio=float(Z),exact_global_bound_fraction_isotropic_anchors=.5,laplace_partition_normalization_error=float(Z-1),weighted_rms_deg=float(a['weights']@np.array([x['rms_deg'] for x in v])),max_shift_deg=max(x['delta_deg'] for x in v)))
(D/'well_averages.json').write_text(json.dumps(averages,indent=2)+'\n');print(json.dumps(averages,indent=2))
