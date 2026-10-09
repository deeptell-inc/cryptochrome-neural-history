"""Extra storage under protected waiting states, with angular-trace memory separated."""
from pathlib import Path
import sys,json
import numpy as np
from scipy import sparse as sp
from scipy.linalg import eig,eigh,solve,lstsq,null_space
from scipy.sparse.linalg import splu
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl.hq_rotation import *
D=ROOT/'data/rotation-cases'
with threadpool_limits(limits=1):
 f=frozen();a=angular_basis(3);na=16;tau=21.476595751657293e-9
 loss=1e7*np.kron(s.PS4,np.eye(9))+4e6*np.eye(36)
 decay=sp.kron(sp.eye(na),.5*(sp.kron(sp.eye(36),loss)+sp.kron(loss.T,sp.eye(36))),format='csc')
 A=(decay-generator(f,a,'RP',tau)).conj().T.tocsc();A0=(decay-generator(f,a,'RP',tau,0.)).conj().T.tocsc()
 slices=[slice(l*l*1296,(l+1)**2*1296) for l in range(4)];lus=[splu(A0[sl,sl].tocsc()) for sl in slices]
 def pre(x):
  z=np.empty_like(x)
  for sl,lu in zip(slices,lus):z[sl]=lu.solve(x[sl])
  return z
 rhs=np.zeros(na*1296,complex);rhs[:1296]=s.vec(1e7*np.kron(s.PS4,np.eye(9)))
 beta=np.linalg.norm(pre(rhs));V=np.zeros((len(rhs),81),complex);V[:,0]=pre(rhs)/beta;HH=np.zeros((81,80),complex)
 for j in range(80):
  w=pre(A@V[:,j])
  for repeat in range(2):
   h=np.einsum('ij,i->j',V[:,:j+1].conj(),w);HH[:j+1,j]+=h;w-=np.einsum('ij,j->i',V[:,:j+1],h)
  HH[j+1,j]=np.linalg.norm(w);V[:,j+1]=w/HH[j+1,j]
  b=np.zeros(j+2,complex);b[0]=beta;y=lstsq(HH[:j+2,:j+1],b)[0];out=np.einsum('ij,j->i',V[:,:j+1],y)
  residual=float(np.linalg.norm(A@out-rhs)/np.linalg.norm(rhs))
  if residual<1e-12:break
 assert residual<1e-12
 inj=s.linear_map(lambda x:np.kron(s.TRIPLET,x),9,36)
 read=np.einsum('ij,ai->aj',inj.conj(),out.reshape(na,1296))
 state=np.load(D/'molecular_L3_protect_waits.npz')['state'];dx=state[:,:,0]-state[:,:,2]
 traces=np.einsum('i,ai->a',s.trvec(9),dx);traces[0]=0.
 scalar=traces[:,None]*s.X0[None,:]
 spin=dx-scalar;spin[0]-=s.X0*np.einsum('i,i->',s.trvec(9),spin[0])
 T=sp.kron(sp.eye(na),sp.csr_matrix(null_space(s.trvec(9)[None,:])),format='csc')
 g=generator(f,a,'H',np.inf);vals,U=eig((T.conj().T@g@T).toarray());coeff=solve(U,T.conj().T@spin.ravel())
 row=(read.conj().ravel()@T);weights=np.einsum('i,ij,j->j',row,U,coeff)
 angular_only=float(np.vdot(read,scalar).real)
 def value(t):return float(np.sum(weights*np.exp(vals*t)).real)+angular_only
 zero=value(0.);reference=json.loads((D/'molecular_L3_protect_waits.json').read_text())['delta_yield'];assert abs(zero-reference)<1e-10
 times=[0.,.012,.2,1.,2.,5.,10.];values=[value(t) for t in times]
 # First half crossing, if present, without assuming an exact single exponential.
 grid=np.r_[0.,np.geomspace(1e-5,30.,250)];half=None
 for lo,hi in zip(grid[:-1],grid[1:]):
  if (value(lo)-zero/2)*(value(hi)-zero/2)<0:half=float(brentq(lambda t:value(t)-zero/2,lo,hi));break
 # Independent orientational-quadrature population propagation for this
 # slow-wait protected case; its agreement is tested, not presumed globally.
 dxnodes=np.einsum('qa,ai->qi',a['Y'],dx);readnodes=np.einsum('qa,ai->qi',a['Y'],read)
 mode_weights=[];mode_rates=[]
 for index,n in enumerate(a['n']):
  h=f['body']['H']+50e-6*sum(n[k]*f['Z']['H'][k] for k in range(3));_,u=eigh(h)
  C=np.column_stack([s.vec(np.outer(u[:,k],u[:,k].conj())) for k in range(9)])
  G=np.einsum('ia,ij,jb->ab',C.conj(),f['D']['H'],C).real
  ev,Uq=eigh((G+G.T)/2);pop=np.einsum('ij,i->j',C.conj(),dxnodes[index]);ell=np.einsum('i,ij->j',readnodes[index].conj(),C)
  weight=np.einsum('i,ij,jk,k->j',ell,Uq,Uq.T,pop)*a['weights'][index]
  mode_weights.append(weight);mode_rates.append(ev)
 mode_weights=np.array(mode_weights);mode_rates=np.array(mode_rates)
 def popvalue(t):return float(np.sum(mode_weights*np.exp(mode_rates*t)).real)
 population_errors=[abs(value(t)-popvalue(t)) for t in times];assert max(population_errors)<1e-9
 half_population=float(brentq(lambda t:popvalue(t)-popvalue(0)/2,0.,10.))
 result=dict(independent_population_max_error=max(population_errors),independent_population_half_s=half_population,extra_delay=[dict(delay_s=t,delta_yield=y) for t,y in zip(times,values)],half_decay_s=half,angular_trace_only_contribution=angular_only,initial_delta=zero,readout_residual=residual,largest_HQ_spin_eigenvalue_real=float(vals.real.max()),interpretation='state-dependent immobilization; no measured binding or neuronal lifetime')
 (D/'protected_wait_storage.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
