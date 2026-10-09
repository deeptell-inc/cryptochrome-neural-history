"""Saved-state decomposition and independent adjoint reaction readout."""
from pathlib import Path
import sys,json,argparse
import numpy as np
from scipy.linalg import lu_solve, null_space, solve
from scipy.sparse.linalg import LinearOperator,gmres
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl.hq_rotation import frozen,ad,s,apply_local
from impl.hq_binding import BindingCycle
p=argparse.ArgumentParser();p.add_argument('--angle',type=float,default=1.);args=p.parse_args();D=ROOT/'data/binding-motion'
with threadpool_limits(limits=1):
 c=BindingCycle(frozen(),2,rms_deg=args.angle);results={}
 for path in sorted(D.glob(f'L2_angle{args.angle:g}_tau1e-11_*.json')):
  d=json.loads(path.read_text());c.configure(d['fraction'],d['koff_s'],d['bias'],build_wait=False)
  z=np.load(path.with_suffix('.npz'));x=(z['free'],z['bound'])
  motion=c.yields(tuple(apply_local(s.RESET,v[:,:,:1]) for v in x))[0].real
  results[d['label']]=dict(instantaneous_spin_contribution=d['yields']['full']-motion,
    motion_distribution_contribution=motion-d['yields']['reset'],
    observed_bound_fractions=np.einsum('q,i,qik->k',c.w,s.trvec(9),x[1]).real.tolist())
  if d['fraction']==.5 and d['koff_s']==0. and not d['bias']:
   # Independent fixed-anchor nuclear cycle matrices, raised to the 20th power.
   Ts={dim:null_space(s.trvec(dim)[None,:]) for dim in (9,18)};values=[]
   for q in range(c.nq):
    waits={}
    for st,rate in [('H',10.),('R',10.),('E',4.)]:
     dim=len(c.f['body'][st]);T=Ts[dim];g=c.gb[st][q];tr=s.trvec(dim)
     red=T.conj().T@g@T
     waits[st]=np.outer(tr,tr)/dim+T@solve(rate*np.eye(dim*dim-1)-red,rate*T.conj().T)
    rr=lu_solve(c.brp[q],c.inj);pm=c.product@rr;em=c.escape@rr
    cyc=waits['H']@waits['R']@(waits['R']@pm+s.TRACE_E@waits['E']@em)
    prepared=np.linalg.matrix_power(cyc,20)@s.X0
    values.append(np.einsum('i,ij,j->',s.trvec(9),pm,prepared-s.X0))
   value=.5*np.dot(c.w,values);err=abs(value.real-d['delta_yield']);assert err<1e-8,(err,value)
   results['independent_static_cycle']=dict(label=d['label'],delta_yield=float(value.real),difference=float(err))
  if d['fraction']==.9 and d['koff_s']==100. and not d['bias']:
   # Adjoint uses a different Krylov solver and observable propagation rather
   # than production forward Neumann iteration and product-state mapping.
   dx=tuple(v[:,:,:1]-v[:,:,2:3] for v in x);inj=tuple(apply_local(c.inj,v)[:,:,0] for v in dx)
   b=c.wobble['RP'];base=-1j*ad(b['body']).toarray()+b['D']
   zz=[ad(v).toarray() for v in c.f['Z']['RP']]
   loss=1e7*np.kron(s.PS4,np.eye(9))+4e6*np.eye(36)
   lossop=.5*(np.kron(np.eye(36),loss)+np.kron(loss.T,np.eye(36)))
   Ab=np.array([lossop-base+1j*c.field*sum(n[a]*zz[a] for a in range(3)) for n in c.angular['n']])
   Af=c.afrp.A*c.afrp.scale;nf=c.na*1296;nb=c.nq*1296
   def unpack(v):return v[:nf].reshape(c.na,1296),v[nf:].reshape(c.nq,1296)
   def pack(a,b):return np.concatenate([a.ravel(),b.ravel()])
   def adj(v):
    a,b=unpack(v);p=np.einsum('qa,ai->qi',c.P.conj().T,a)-b
    fa=(Af.conj().T@a.ravel()).reshape(a.shape)+np.einsum('aq,q,qi->ai',c.Y.conj().T,c.capture,p)
    bb=np.einsum('qji,qj->qi',Ab.conj(),b)-c.koff*p
    return pack(fa,bb)
   def pre(v):
    a,b=unpack(v);fa=c.afrp.lu.solve(a.ravel(),trans='H').reshape(a.shape)/c.afrp.scale
    bb=np.array([lu_solve(lu,rhs,trans=2) for lu,rhs in zip(c.brp,b)])
    return pack(fa,bb)
   obs=s.vec(1e7*np.kron(s.PS4,np.eye(9)));of=np.zeros((c.na,1296),complex);of[0]=obs
   ob=c.w[:,None]*obs[None,:];rhs=pack(of,ob)
   op=LinearOperator((nf+nb,nf+nb),matvec=adj,dtype=complex);pr=LinearOperator(op.shape,matvec=pre,dtype=complex)
   yy,info=gmres(op,rhs,M=pr,rtol=2e-12,atol=0,restart=12,maxiter=12);assert info==0,info
   value=np.vdot(yy,pack(*inj));err=abs(value.real-d['delta_yield'])
   residual=np.linalg.norm(adj(yy)-rhs)/np.linalg.norm(rhs);assert err<1e-10 and residual<1e-10,(err,residual)
   results['independent_adjoint']=dict(label=d['label'],delta_yield=float(value.real),difference=float(err),residual=float(residual))
   del Ab
 (D/f'augment_angle{args.angle:g}.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results),flush=True)
