from pathlib import Path
import sys,json,itertools
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl.hq_field_couplings import well
D=ROOT/'data/field-couplings';rows=[]
for deg in (1.,5.):
 for angle in (0.,45.,90.,135.,180.):
  n=np.array([np.sin(np.deg2rad(angle)),0,np.cos(np.deg2rad(angle))]);k=10.910941626120245;w=well(n,deg,1e-11,k)
  previous=None
  for order in (7,11):
   nodes,weights=np.polynomial.hermite.hermgauss(order);points=np.array(list(itertools.product(range(order),repeat=3)))
   rv=nodes[points]*np.sqrt(2/w['c']);weights3=np.prod(weights[points]/np.sqrt(np.pi),axis=1);R=Rotation.from_rotvec(rv).as_matrix()
   norm=np.linalg.norm(rv,axis=1);haar=np.sinc(norm/(2*np.pi))**2
   energy_e=-k*np.einsum('a,na->n',n,R[:,:,2]);boltz=weights3*haar*np.exp(-energy_e-k)
   Z0=np.sum(weights3*haar);Z=np.sum(boltz)*np.exp(k);q=boltz/boltz.sum()
   tangent=Rotation.from_matrix(np.einsum('nij,jk->nik',R,w['R'].T)).as_rotvec();mu=q@tangent
   cv=np.einsum('n,ni,nj->ij',q,tangent-mu,tangent-mu)
   expected=np.einsum('ij,j,kj->ik',w['axes'],w['variances'],w['axes'])
   out=dict(rms0_deg=deg,anchor_angle_deg=angle,quadrature_order=order,tangent_mean_norm_deg=float(np.rad2deg(np.linalg.norm(mu))),covariance_relative_error=float(np.linalg.norm(cv-expected)/np.linalg.norm(cv)),exact_F_res_kBT=float(-np.log(Z/Z0)+k*n[2]),laplace_F_res_kBT=w['free_energy_residual'])
   if previous is not None:out['quadrature_covariance_relative_change']=float(np.linalg.norm(cv-previous)/np.linalg.norm(cv))
   previous=cv;rows.append(out)
(D/'well_independent.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
