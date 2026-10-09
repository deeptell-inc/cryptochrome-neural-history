from pathlib import Path
import sys,json
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import eigs
from scipy.linalg import eigvals
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl.hq_rotation import frozen,s
from impl.hq_electric import electric_basis
from impl.hq_field_couplings import matched_h_response,coupled_generator
D=ROOT/'data/field-couplings';rows=[]
with threadpool_limits(limits=1):
 f=frozen();response,_=matched_h_response()
 for field in (0.,14.,-14.):
  for L in (2,3,4):
   a=electric_basis(L,10.910941626120245);g=coupled_generator(response,field)(f,a,'H',21.476595751657293e-9)
   ident=np.zeros(g.shape[0],complex);ident[:81]=s.trvec(9)/3;g-=1e8*sp.csc_matrix(np.outer(ident,ident.conj()))
   ev=eigs(g,k=5,sigma=0,tol=1e-11,v0=np.random.default_rng(503).normal(size=len(ident)),return_eigenvectors=False);rate=-ev.real.max()
   row=dict(field_MVm=field,cutoff=L,rate_s=float(rate),inverse_slowest_us=float(1e6/rate),orienting_kappa_fixed=10.910941626120245)
   if L==2 and field==14.:
    alternate=-eigvals(g.toarray()).real.max();row['independent_relative_error']=float(abs(alternate-rate)/rate)
   rows.append(row)
 (D/'HQ_spectrum.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
