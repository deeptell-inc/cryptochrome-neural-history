"""Independent GMRES solution with B=0 block factors, for reset-readout convergence.
This validates the reaction observable; it is not a second physical model.
"""
from pathlib import Path
import sys,json,argparse,time
import numpy as np
from scipy.linalg import lstsq
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl.hq_rotation import *
p=argparse.ArgumentParser();p.add_argument('--cutoff',type=int,default=4);p.add_argument('--tau-ns',type=float,default=99.12);args=p.parse_args()
with threadpool_limits(limits=1):
 start=time.monotonic();f=frozen();a=angular_basis(args.cutoff);na=len(a['lm'])
 loss=1e7*np.kron(s.PS4,np.eye(9))+4e6*np.eye(36)
 decay=sp.kron(sp.eye(na),.5*(sp.kron(sp.eye(36),loss)+sp.kron(loss.T,sp.eye(36))),format='csc')
 A=decay-generator(f,a,'RP',args.tau_ns*1e-9);A0=decay-generator(f,a,'RP',args.tau_ns*1e-9,0.)
 lus=[];slices=[]
 for l in range(args.cutoff+1):
  sl=slice(l*l*1296,(l+1)**2*1296);slices.append(sl);lus.append(splu(A0[sl,sl].tocsc()))
  print(json.dumps(dict(l=l,nnz=lus[-1].nnz,seconds=time.monotonic()-start)),flush=True)
 def precondition(x):
  out=np.empty_like(x)
  for sl,lu in zip(slices,lus):out[sl]=lu.solve(x[sl])
  return out
 rhs=np.zeros(na*1296,complex);rhs[:1296]=s.vec(np.kron(s.TRIPLET,np.eye(9)/9));iters=[]
 # Explicit Arnoldi contractions avoid the environment's NumPy matmul warnings.
 beta=np.linalg.norm(precondition(rhs));V=np.zeros((len(rhs),81),complex);V[:,0]=precondition(rhs)/beta
 H=np.zeros((81,80),complex);v=np.zeros_like(rhs);converged=False
 for j in range(80):
  w=precondition(A@V[:,j])
  for repeat in range(2):
   h=np.einsum('ij,i->j',V[:,:j+1].conj(),w);H[:j+1,j]+=h
   w-=np.einsum('ij,j->i',V[:,:j+1],h)
  H[j+1,j]=np.linalg.norm(w);V[:,j+1]=w/H[j+1,j]
  bsmall=np.zeros(j+2,complex);bsmall[0]=beta
  y=lstsq(H[:j+2,:j+1],bsmall)[0];v=np.einsum('ij,j->i',V[:,:j+1],y)
  rr=float(np.linalg.norm(A@v-rhs)/np.linalg.norm(rhs));iters.append(rr)
  if rr<1e-12:converged=True;break
 assert converged

 residual=float(np.linalg.norm(A@v-rhs)/np.linalg.norm(rhs));assert residual<2e-12
 rho=s.unvec(v[:1296],36);observable=1e7*np.kron(s.PS4,np.eye(9));y=np.einsum('ij,ji->',observable,rho)
 angular_states=np.einsum('qa,ai->qi',a['Y'],v.reshape(na,1296))
 mineig=1.;hermiticity=0.
 for state in angular_states:
  matrix=s.unvec(state,36);weight=np.trace(matrix).real
  hermiticity=max(hermiticity,float(np.linalg.norm(matrix-matrix.conj().T)/np.linalg.norm(matrix)))
  mineig=min(mineig,float(np.linalg.eigvalsh((matrix+matrix.conj().T)/2).min()/weight))
 assert mineig>-1e-9 and hermiticity<1e-9
 out=dict(minimum_conditional_RP_eigenvalue=mineig,RP_relative_hermiticity_error=hermiticity,cutoff=args.cutoff,tau_ns=args.tau_ns,yield_reset_uniform=float(y.real),imag=float(y.imag),relative_residual=residual,iterations=len(iters),seconds=time.monotonic()-start)
 (ROOT/f'data/hq-rotation/independent_L{args.cutoff}_tau{args.tau_ns:.8g}.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
