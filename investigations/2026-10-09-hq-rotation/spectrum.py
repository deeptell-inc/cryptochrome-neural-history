"""HQ decay spectrum at the matched field; no interpretation as measured T1."""
from pathlib import Path
import sys,json,time
import numpy as np
from scipy.linalg import eigvals
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl.hq_rotation import *
with threadpool_limits(limits=1):
 f=frozen();rows=[]
 for cutoff in [2,3,4]:
  angular=angular_basis(cutoff)
  for tau in [9.72,21.476595751657293,99.12]:
   g=generator(f,angular,'H',tau*1e-9).toarray()
   # Remove the known constant identity with an orthonormal traceless-spin basis.
   # Nonconstant scalar harmonics are included; they decay by rotational diffusion.
   from scipy.linalg import null_space
   ident=np.zeros(len(g),complex);ident[:81]=s.trvec(9)/3
   T=null_space(ident[None,:]);red=np.einsum('ia,ij,jb->ab',T.conj(),g,T,optimize=True)
   ev=eigvals(red);rate=float(-ev.real.max())
   assert rate>0
   from scipy.sparse.linalg import eigs
   stabilized=sp.csc_matrix(g)-1e8*sp.csc_matrix(np.outer(ident,ident.conj()))
   independent=eigs(stabilized,k=3,sigma=0,return_eigenvectors=False,tol=1e-11)
   alternate=float(-independent.real.max());relative_error=abs(alternate-rate)/rate
   assert relative_error<1e-8
   rows.append(dict(cutoff=cutoff,tau_ns=tau,slowest_rate_s=rate,inverse_slowest_rate_us=1e6/rate,shift_invert_rate_s=alternate,independent_relative_error=relative_error,log10_slowest_exponential_at_12ms=-rate*.012/np.log(10)))
   print(json.dumps(rows[-1]),flush=True)
 (ROOT/'data/hq-rotation/HQ_spectrum.json').write_text(json.dumps(rows,indent=2)+'\n')
