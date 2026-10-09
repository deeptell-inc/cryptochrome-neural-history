"""Independent projected waiting solves and Arnoldi reaction readout checks."""
from pathlib import Path
import sys,json
import numpy as np
from scipy import sparse as sp
from scipy.linalg import null_space,solve,lstsq
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl.hq_rotation import frozen,angular_basis,generator,s,LinearSolve
D=ROOT/'data/rotation-cases';report={}
with threadpool_limits(limits=1):
 f=frozen();angular=angular_basis(3);na=16;tau=21.476595751657293e-9
 T=sp.kron(sp.eye(na),sp.csr_matrix(null_space(s.trvec(9)[None,:])),format='csc')
 rng=np.random.default_rng(261009);u=rng.normal(size=(9,9))+1j*rng.normal(size=(9,9));rho=np.einsum('ij,kj->ik',u,u.conj());rho/=np.trace(rho)
 rhs=np.zeros((na,81),complex);rhs[0]=s.vec(rho)-s.X0;rhs=rhs.ravel()
 for rate,mobility in [(10.,tau),(1e5,tau),(1e7,tau),(1e9,tau),(10.,np.inf)]:
  g=generator(f,angular,'H',mobility)
  A=rate*sp.eye(na*81,format='csc')-g
  projected=(T.conj().T@A@T).toarray();z=T@solve(projected,rate*(T.conj().T@rhs))
  lifted=A+1e8*sp.kron(sp.eye(na),np.outer(s.trvec(9),s.trvec(9))/9,format='csc')
  other=splu(lifted).solve(rate*rhs).reshape(na,81)
  other-=s.X0[None,:]*np.einsum('i,ai->a',s.trvec(9),other)[:,None]
  error=float(np.linalg.norm(z-other.ravel()));assert error<1e-9
  report[f'H_wait_{rate:g}_{mobility:g}']=error
 loss=1e7*np.kron(s.PS4,np.eye(9))+4e6*np.eye(36)
 decay=sp.kron(sp.eye(na),.5*(sp.kron(sp.eye(36),loss)+sp.kron(loss.T,sp.eye(36))),format='csc')
 A=decay-generator(f,angular,'RP',tau);A0=decay-generator(f,angular,'RP',tau,0.)
 slices=[slice(l*l*1296,(l+1)**2*1296) for l in range(4)];lus=[splu(A0[sl,sl].tocsc()) for sl in slices]
 def pre(x):
  y=np.empty_like(x)
  for sl,lu in zip(slices,lus):y[sl]=lu.solve(x[sl])
  return y
 def arnoldi(rhs):
  beta=np.linalg.norm(pre(rhs));V=np.zeros((len(rhs),81),complex);V[:,0]=pre(rhs)/beta;H=np.zeros((81,80),complex)
  for j in range(80):
   w=pre(A@V[:,j])
   for repeat in range(2):
    h=np.einsum('ij,i->j',V[:,:j+1].conj(),w);H[:j+1,j]+=h;w-=np.einsum('ij,j->i',V[:,:j+1],h)
   H[j+1,j]=np.linalg.norm(w);V[:,j+1]=w/H[j+1,j]
   b=np.zeros(j+2,complex);b[0]=beta;y=lstsq(H[:j+2,:j+1],b)[0];out=np.einsum('ij,j->i',V[:,:j+1],y)
   residual=float(np.linalg.norm(A@out-rhs)/np.linalg.norm(rhs))
   if residual<1e-12:return out,residual
  raise RuntimeError('no convergence')
 inj=s.linear_map(lambda x:np.kron(s.TRIPLET,x),9,36);obs=1e7*np.kron(s.PS4,np.eye(9))
 for label in ['protect_waits','all_fast_10000','all_fast_1e+06','all_fast_1e+08']:
  x=np.load(D/f'molecular_L3_{label}.npz')['state'];dx=x[:,:,0]-x[:,:,2]
  # Remove numerical mass drift before testing the tiny observable.
  dx[0]-=s.X0*np.einsum('i,i->',s.trvec(9),dx[0])
  rhs=np.einsum('ij,aj->ai',inj,dx).ravel();v,res=arnoldi(rhs)
  val=float(np.einsum('ij,ji->',obs,s.unvec(v[:1296],36)).real)
  reference=json.loads((D/f'molecular_L3_{label}.json').read_text())['delta_yield']
  err=abs(reference-val);assert err<2e-11
  report[label]=dict(independent_delta=val,reference=reference,error=err,residual=res)
  print(json.dumps({label:report[label]}),flush=True)
 report['status']='passed';(D/'molecular_independent_checks.json').write_text(json.dumps(report,indent=2)+'\n')
