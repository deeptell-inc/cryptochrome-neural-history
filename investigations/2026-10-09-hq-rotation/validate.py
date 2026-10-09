"""Independent limiting and frozen-reference checks; assertions are scientific gates."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.linalg import null_space, solve, eigh
from scipy import sparse as sp
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl.hq_rotation import *
mm=lambda a,b:np.einsum('ij,jk->ik',a,b,optimize=False)
mv=lambda a,b:np.einsum('ij,j->i',a,b,optimize=False)
def wait(L,r):
 d=round(np.sqrt(len(L)));t=s.trvec(d);T=null_space(t[None,:]);reduced=mm(T.conj().T,mm(L,T))
 return np.outer(t,t)/d+mm(T,solve(r*np.eye(len(reduced))-reduced,r*T.conj().T))
with threadpool_limits(limits=1):
 f=frozen();a=angular_basis(2);report={}
 report['angular_orthogonality']=float(np.max(abs(mm(a['inverse'],a['Y'])-np.eye(9))))
 # A laboratory-fixed vector n.J must be annihilated by L_orb + ad J.
 for state in ['H','E']:
  d=len(f['H'][state]);J=f['spin'][state]
  v=sum(a['N'][axis].toarray()[:,0,None]*s.vec(J[axis])[None,:] for axis in range(3)).ravel()
  errors=[]
  for axis in range(3):
   K=sp.kron(a['L'][axis],sp.eye(d*d))+sp.kron(sp.eye(9),ad(J[axis]))
   errors.append(float(np.linalg.norm(K@v)/np.linalg.norm(v)))
  report[state+'_lab_vector_covariance']=max(errors)
  assert max(errors)<1e-12
 # Rebuild clamped RP resolvent using archived local Hamiltonian/covariance.
 L=-1j*ad(f['H']['RP'])+sp.csr_matrix(f['D']['RP'])
 loss=1e7*np.kron(s.PS4,np.eye(9))+4e6*np.eye(36)
 A=-L+.5*(sp.kron(sp.eye(36),loss)+sp.kron(loss.T,sp.eye(36)))
 inj=s.linear_map(lambda x:np.kron(s.TRIPLET,x),9,36)
 bra=np.kron(s.SINGLET[None,:],np.eye(9))
 product=s.linear_map(lambda x:1e7*np.einsum('ai,ij,bj->ab',bra,x,bra.conj()),36,9)
 escape=s.linear_map(lambda x:4e6*s.escape_state(x),36,18)
 v=splu(A.tocsc()).solve(inj);p=mm(product,v);e=mm(escape,v)
 for name,value in [('P_0',p),('Escape_0',e)]:
  report[name+'_relative_error']=float(np.linalg.norm(value-f['archive'][name])/np.linalg.norm(f['archive'][name]))
  assert report[name+'_relative_error']<1e-10
 w={st:wait(-1j*ad(f['H'][st]).toarray()+f['D'][st],r) for st,r in [('H',10.),('R',10.),('E',4.)]}
 M=mm(w['H'],mm(w['R'],mm(w['R'],p)+mm(s.TRACE_E,mm(w['E'],e))))
 _,u=eigh(f['H']['H']);C=np.column_stack([s.vec(np.outer(u[:,j],u[:,j].conj())) for j in range(9)]);DP=mm(C,C.conj().T)
 ys={}
 for name,update in [('full',M),('population',mm(DP,mm(M,DP))),('reset',mm(s.RESET,mm(M,s.RESET)))]:
  x=s.X0.copy()
  for i in range(20):x=mv(update,x)
  ys[name]=float(np.einsum('i,i->',s.trvec(9),mv(p,x)).real)
 report['clamped_yields']=ys
 for name,target in [('full',.24408853990175508),('population',.24408853989235246),('reset',.23838660744880275)]:
  assert abs(ys[name]-target)<1e-9
 # B=0, l=0 reduces exactly to common spin-rotation Lindbladian.
 for state in ['H','E']:
  tau=21.476595751657293e-9
  direct=-1j*ad(f['body'][state]).toarray()+f['D'][state]+sum(dense_dissipator(j)/(3*tau) for j in f['spin'][state])
  g=generator(f,angular_basis(0),state,tau,0).toarray()
  err=float(np.linalg.norm(g-direct)/np.linalg.norm(direct));report[state+'_B0_reduction_error']=err;assert err<1e-12
 report['status']='passed'
 (ROOT/'data/hq-rotation/validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
