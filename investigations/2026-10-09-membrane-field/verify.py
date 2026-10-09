from pathlib import Path
import sys,json
import numpy as np
from scipy import sparse as sp
from scipy.linalg import null_space
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl.hq_rotation import frozen,s
from impl.hq_electric import electric_basis,electric_generator
from impl.hq_binding import BindingCycle
D=ROOT/'data/membrane-field';report={};mul=lambda a,b:np.einsum('ij,jk->ik',a,b,optimize=False)
with threadpool_limits(limits=1):
 f=frozen();T=null_space(s.trvec(9)[None,:]);rng=np.random.default_rng(261009501)
 a=electric_basis(2,10.910941626120245)
 c=BindingCycle(f,2,rms_deg=1.,build_reaction=False,angular_override=a,generator_override=electric_generator)
 for off in (1.,100.):
  c.configure(.5,off,0.);na,nq=c.na,c.nq
  F=sp.kron(sp.eye(na),T,format='csc');B=sp.kron(sp.eye(nq),T,format='csc')
  gf=F.conj().T@c.gf['H']@F;gb=sp.block_diag([mul(mul(T.conj().T,g),T) for g in c.gb['H']],format='csc')
  cap=sp.kron(sp.csr_matrix(c.capture[:,None]*c.Y),sp.eye(80),format='csc');rel=off*sp.kron(sp.csr_matrix(c.P),sp.eye(80),format='csc')
  G=sp.bmat([[gf-sp.kron(c.mult,sp.eye(80)),rel],[cap,gb-off*sp.eye(nq*80)]],format='csc');A=10*sp.eye(G.shape[0],format='csc')-G
  u=rng.normal(size=(9,9))+1j*rng.normal(size=(9,9));rho=mul(u,u.conj().T);rho/=np.trace(rho)
  x=tuple(v[:,:,0:1].copy() for v in c.initial());x[0][0,:,0]=.5*s.vec(rho);x[1][:,:,0]=.5*s.vec(rho)
  y=c.wait('H',x,10.);rhs=np.concatenate([F.conj().T@x[0].ravel(),B.conj().T@x[1].ravel()]);direct=splu(A).solve(10*rhs)
  ref=np.concatenate([F.conj().T@y[0].ravel(),B.conj().T@y[1].ravel()]);err=float(np.linalg.norm(direct-ref));assert err<2e-8
  report[f'direct_H_off{off:g}']=dict(state_error=err,trace_error=float(max(abs(c.total_trace(y)-1))),residual=float(np.linalg.norm(A@direct-10*rhs)/np.linalg.norm(10*rhs)))
 # Independent uniform Legendre integration verifies weighted quadrature moments.
 zz,ww=np.polynomial.legendre.leggauss(300);kk=a['kappa'];ww*=np.exp(kk*zz-kk);ww/=ww.sum()
 report['moments']=dict(mean_cos=float(a['weights']@a['n'][:,2]),P2=float(a['weights']@((3*a['n'][:,2]**2-1)/2)),reference_mean_cos=float(ww@zz),reference_P2=float(ww@((3*zz**2-1)/2)))
 # A higher positive quadrature must reproduce weak operator matrices.
 checks=[]
 for axis in range(3):
  a=electric_basis(3,kk,axis);b=electric_basis(3,kk,axis,nz=12)
  ga=electric_generator(f,a,'H',21.476595751657293e-9);gb=electric_generator(f,b,'H',21.476595751657293e-9)
  checks.append(dict(axis=axis,generator_relative_error=float(sp.linalg.norm(ga-gb)/sp.linalg.norm(ga))))
 report['angular_quadrature']=checks
 # Frozen HQ thermal distribution is only an energy-scale diagnostic, not E-induced preparation.
 from scipy.constants import hbar,k
 from scipy.linalg import eigh
 ev,u=eigh(f['H']['H']);p=np.exp(-hbar*(ev-ev.min())/(k*310));p/=p.sum()
 report['HQ_thermal_energy_scale']=dict(max_population_deviation=float(max(abs(p-1/9))),trace_distance_identity=float(sum(abs(p-1/9))/2),source='frozen HQ Hamiltonian; not membrane-induced polarization')
 (D/'independent_checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
