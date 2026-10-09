from pathlib import Path
import sys,json,time
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import eigs
from scipy.linalg import eigvals
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl.hq_rotation import frozen,s
from impl.hq_electric import electric_basis,electric_generator
D=ROOT/'data/membrane-field';rows=[]
with threadpool_limits(limits=1):
 f=frozen()
 for axis in [2,0,1]:
  for k in ([0.,.10910941626120247,1.0910941626120247,10.910941626120245] if axis==2 else [1.0910941626120247,10.910941626120245]):
   for L in [2,3,4]:
    t=time.monotonic();a=electric_basis(L,k,axis);g=electric_generator(f,a,'H',21.476595751657293e-9)
    ident=np.zeros(g.shape[0],complex);ident[:81]=s.trvec(9)/3
    stabilized=g-1e8*sp.csc_matrix(np.outer(ident,ident.conj()))
    ev,vec=eigs(stabilized,k=5,sigma=0,tol=1e-11,v0=np.random.default_rng(501).normal(size=len(ident)))
    idx=np.argmax(ev.real);rate=-ev[idx].real
    residual=np.linalg.norm(stabilized@vec[:,idx]-ev[idx]*vec[:,idx])/max(np.linalg.norm(stabilized@vec[:,idx]),1)
    r=dict(axis=axis,kappa=k,cutoff=L,slowest_rate_s=rate,inverse_slowest_us=1e6/rate,log10_exponential_at_12ms=-rate*.012/np.log(10),eigen_residual=float(residual),seconds=time.monotonic()-t)
    if L==2 and axis==2 and k in (0.,10.910941626120245):
     dense=-eigvals(stabilized.toarray()).real.max();r.update(dense_rate_s=float(dense),independent_relative_error=float(abs(dense-rate)/rate))
    rows.append(r);(D/'HQ_spectrum.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(r),flush=True)
