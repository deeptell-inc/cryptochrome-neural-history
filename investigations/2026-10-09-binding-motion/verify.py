"""Independent trace-free coupled solve and finite colored-noise comparison."""
from pathlib import Path
import sys,json,itertools
import numpy as np
from scipy import sparse as sp
from scipy.linalg import null_space,solve,expm,hadamard,eigvals
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl.hq_rotation import frozen,ad,s
from impl.hq_binding import BindingCycle,wobble_operators
D=ROOT/'data/binding-motion';report={}
with threadpool_limits(limits=1):
 f=frozen();T=null_space(s.trvec(9)[None,:]);rng=np.random.default_rng(261009)
 c=BindingCycle(f,2,rms_deg=1.,build_reaction=False)
 for off,bias in [(0.,0.),(1.,0.),(100.,1.8)]:
  c.configure(.5,off,bias);na,nq=c.na,c.nq
  # An independent direct solve of BOTH mobility blocks, without the production
  # Schur fixed-point iteration, and with identity removed by an explicit basis.
  F=sp.kron(sp.eye(na),T,format='csc');B=sp.kron(sp.eye(nq),T,format='csc')
  gf=F.conj().T@c.gf['H']@F;gb=sp.block_diag([T.conj().T@g@T for g in c.gb['H']],format='csc')
  cap=sp.kron(sp.csr_matrix(c.capture[:,None]*c.Y),sp.eye(80),format='csc')
  rel=off*sp.kron(sp.csr_matrix(c.P),sp.eye(80),format='csc')
  generator=sp.bmat([[gf-sp.kron(c.mult,sp.eye(80)),rel],[cap,gb-off*sp.eye(nq*80)]],format='csc')
  A=10*sp.eye(generator.shape[0],format='csc')-generator
  # Isotropic random spin state with identical conditional spin in both pools.
  u=rng.normal(size=(9,9))+1j*rng.normal(size=(9,9));rho=u@u.conj().T;rho/=np.trace(rho)
  x=c.initial();x=tuple(v[:,:,0:1].copy() for v in x)
  spin=s.vec(rho);x[0][0,:,0]=.5*spin;x[1][:,:,0]=.5*c.profile[:,None]*spin
  y=c.wait('H',x,10.)
  rhs=np.concatenate([np.asarray(F.conj().T@x[0].ravel()),np.asarray(B.conj().T@x[1].ravel())])
  direct=splu(A).solve(10*rhs);ref=np.concatenate([F.conj().T@y[0].ravel(),B.conj().T@y[1].ravel()])
  error=float(np.linalg.norm(direct-ref));assert error<2e-8,error
  tr=float(max(abs(c.total_trace(y)-1)));assert tr<1e-11
  report[f'direct_H_off{off:g}_bias{bias:g}']=dict(state_error=error,trace_error=tr,
      residual=float(np.linalg.norm(A@direct-10*rhs)/np.linalg.norm(10*rhs)))
 # Exact 8-state independent telegraph angles. Each component +/- theta/sqrt(3),
 # flip rate 1/(2*tau), matching the reduced model's variance and autocorrelation.
 # Transform noise modes and eliminate the seven fast modes by a dense Schur
 # complement; avoid subtracting huge stochastic rates from slow spin modes.
 signs=np.array(list(itertools.product([-1,1],repeat=3)));Q=hadamard(8)/np.sqrt(8)
 K=np.zeros((8,8))
 for i in range(8):
  for bit in (1,2,4):K[i^bit,i]=.5
  K[i,i]=-1.5
 diag=np.diag(Q.T@K@Q)
 examples=[]
 for deg,tau in [(0.1,1e-11),(1.,1e-11),(5.,1e-11),(1.,1e-10),(5.,1e-10)]:
  wobble,_=wobble_operators(f,deg,tau);gs=[];sigma=np.deg2rad(deg)/np.sqrt(3)
  for row in signs:
   u=expm(-1j*sigma*sum(row[a]*f['spin']['H'][a] for a in range(3)));su=np.kron(u.conj(),u)
   h=u@f['body']['H']@u.conj().T+50e-6*f['Z']['H'][2]
   g=-1j*ad(h).toarray()+su@f['D']['H']@su.conj().T
   gs.append(T.conj().T@g@T)
  block=np.einsum('ia,ib,ijk->ajbk',Q,Q,np.array(gs)).reshape(640,640)
  block+=np.kron(np.diag(diag/tau),np.eye(80))
  slow=block[:80,:80];fast=block[80:,80:]
  effective=slow+block[:80,80:]@solve(10*np.eye(560)-fast,block[80:,:80])
  reduced=T.conj().T@(-1j*ad(wobble['H']['body']+50e-6*f['Z']['H'][2]).toarray()+wobble['H']['D'])@T
  exact_map=solve(10*np.eye(80)-effective,10*np.eye(80))
  reduced_map=solve(10*np.eye(80)-reduced,10*np.eye(80))
  err=float(np.linalg.norm(exact_map-reduced_map,2))
  relerr=float(np.linalg.norm(exact_map-reduced_map)/np.linalg.norm(exact_map))
  examples.append(dict(rms_deg=deg,tau_w_s=tau,HQ_wait_map_operator_error=err,
                       HQ_wait_map_relative_frobenius_error=relerr))
 report['colored_noise_comparison']=examples
 report['scope']='Independent HQ wait checks, not proof of a native neuronal motion model or exact OU equivalence.'
 (D/'independent_checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
