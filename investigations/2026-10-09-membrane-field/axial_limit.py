from pathlib import Path
import sys,json
import numpy as np
from scipy import sparse as sp
from scipy.linalg import eigvals
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl.hq_rotation import frozen,ad,s
from impl.hq_electric import ElectricCycle
D=ROOT/'data/membrane-field'
def gen(f,a,st,tau,field):
 axis=a['axis'];h=f['body'][st]+field*f['Z'][st][axis];c=ad(f['spin'][st][axis])
 return (-1j*ad(h)+sp.csr_matrix(f['D'][st])-c@c/(6*tau)).tocsc()
with threadpool_limits(limits=1):
 f=frozen();rows=[]
 for axis in range(3):
  a=dict(lm=[(0,0)],n=np.eye(3)[axis:axis+1],Y=np.ones((1,1)),inverse=np.ones((1,1)),weights=np.ones(1),scalar_rotation=np.zeros((1,1)),axis=axis)
  c=ElectricCycle(f,0,0,axis,angular_override=a,generator_override=gen);x,r=c.run()
  ident=s.trvec(9)/3;ev=eigvals(c.generators['H'].toarray()-1e8*np.outer(ident,ident))
  r.update(axis=axis,limit='perfect dipole-axis orientation; unrestricted axial Brownian rotation',inverse_slowest_us=float(-1e6/ev.real.max()))
  rows.append(r);print(json.dumps(r),flush=True)
 (D/'axial_limit.json').write_text(json.dumps(rows,indent=2)+'\n')
