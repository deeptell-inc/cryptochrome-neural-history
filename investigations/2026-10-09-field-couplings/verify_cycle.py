from pathlib import Path
import sys,json
import numpy as np
from scipy import sparse as sp
from scipy.linalg import null_space
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl.hq_rotation import frozen,s
from impl.hq_electric import electric_basis
from impl.hq_binding import BindingCycle
from impl.hq_field_couplings import matched_h_response,coupled_generator,BoundFieldModel,well
D=ROOT/'data/field-couplings';k=10.910941626120245;mul=lambda a,b:np.einsum('ij,jk->ik',a,b,optimize=False)
with threadpool_limits(limits=1):
 f=frozen();response,raw=matched_h_response();a=electric_basis(2,k);model=BoundFieldModel(f,response,14.,k,5.)
 c=BindingCycle(f,2,rms_deg=5.,build_reaction=False,angular_override=a,generator_override=coupled_generator(response,14.),node_model=model.at)
 profile=np.array([np.exp(-well(n,5.,1e-11,k)['free_energy_residual']) for n in a['n']]);profile/=a['weights']@profile;c.configure(.5,1.,0.,profile_override=profile)
 T=null_space(s.trvec(9)[None,:]);na,nq=c.na,c.nq;F=sp.kron(sp.eye(na),T,format='csc');B=sp.kron(sp.eye(nq),T,format='csc')
 gf=F.conj().T@c.gf['H']@F;gb=sp.block_diag([mul(mul(T.conj().T,g),T) for g in c.gb['H']],format='csc');cap=sp.kron(sp.csr_matrix(c.capture[:,None]*c.Y),sp.eye(80),format='csc');rel=sp.kron(sp.csr_matrix(c.P),sp.eye(80),format='csc')
 G=sp.bmat([[gf-sp.kron(c.mult,sp.eye(80)),rel],[cap,gb-sp.eye(nq*80)]],format='csc');A=10*sp.eye(G.shape[0],format='csc')-G
 rng=np.random.default_rng(261009502);u=rng.normal(size=(9,9))+1j*rng.normal(size=(9,9));rho=mul(u,u.conj().T);rho/=np.trace(rho)
 x=tuple(v[:,:,0:1].astype(complex) for v in c.initial());x[0][0,:,0]=.5*s.vec(rho);x[1][:,:,0]=.5*profile[:,None]*s.vec(rho)
 y=c.wait('H',x,10.);rhs=np.concatenate([F.conj().T@x[0].ravel(),B.conj().T@x[1].ravel()]);direct=splu(A).solve(10*rhs);ref=np.concatenate([F.conj().T@y[0].ravel(),B.conj().T@y[1].ravel()]);err=float(np.linalg.norm(direct-ref));assert err<2e-8
 # Verify third-rank response frame law with one explicit saved rotation.
 from impl import dark_md_coefficients as md
 from impl import dark_electronic_tensors as et
 from scipy.constants import physical_constants
 _,Q,_,_,_,_=md.coefficient_series(2026091521,basis='svpd',external=True,tag='_dense');zz=np.load(D/'H/response.npz');chi=zz['dV_au_per_field_au'];ev=np.array([.2,.3,.7]);rot=Q[17,0]
 directV=mul(mul(rot,np.einsum('a,aij->ij',rot.T@ev,chi[0])),rot.T)
 transformed=np.einsum('ib,jc,dbc,ad->aij',rot,rot,chi[0],rot)
 tensor_error=float(np.max(abs(directV-np.einsum('a,aij->ij',ev,transformed))));assert tensor_error<1e-9
 result=dict(direct_H_state_error=err,direct_H_residual=float(np.linalg.norm(A@direct-10*rhs)/np.linalg.norm(10*rhs)),third_rank_transform_error=tensor_error,field_wobble_validity=model.validity)
 (D/'independent_cycle.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
