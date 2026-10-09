"""Matched static spin-electric response and conditional harmonic binding wells.

Electronic susceptibility is transported through the saved local MD rotations
before coupling to the common whole-protein orientation. The old intrinsic
local bath is retained; global and bound-wobble relaxation use the new H(E).
"""
from pathlib import Path
import numpy as np
from scipy import sparse as sp
from scipy.linalg import eigh,expm
from scipy.spatial.transform import Rotation
from scipy.optimize import brentq
from scipy.linalg.blas import zgemm
from .hq_rotation import ad,dense_dissipator,s,frozen
from .hq_electric import electric_generator
from . import dark_md_coefficients as md
from . import dark_electronic_tensors as electronic
ROOT=Path(__file__).resolve().parents[1]


def well(n,rms_deg,tau0,kappa,axis=2):
    """Exact well minimum; tangent harmonic Hessian on SO(3), kBT units.

    u(R)=c |log R|^2/2-kappa n.Rp. Friction is fixed at its zero-field
    value, giving Cov=K^-1 and correlation times c*tau0/eig(K).
    Free-energy shift is local harmonic, not a binding PMF or escape barrier.
    """
    n=np.asarray(n,float);n=n/np.linalg.norm(n);p=np.eye(3)[axis]
    c=3/np.deg2rad(rms_deg)**2
    if abs(kappa)>=c:raise ValueError('requires globally stiff single-well regime |kappa|<c')
    theta=np.arccos(np.clip(n@p,-1,1));cross=np.cross(p,n);norm=np.linalg.norm(cross)
    if norm<1e-12 or kappa==0:delta=0.;v=np.array([1.,0.,0.])
    else:
        v=cross/norm;limit=abs(kappa)/c+1e-12
        delta=brentq(lambda x:c*x-kappa*np.sin(theta-x),-limit,limit,xtol=1e-15)
    rv=delta*v;R=Rotation.from_rotvec(rv).as_matrix();pb=R@p
    fac=1. if abs(delta)<1e-8 else .5*delta/np.tan(.5*delta)
    K=c*(fac*np.eye(3)+(1-fac)*np.outer(v,v))
    K+=kappa*((n@pb)*np.eye(3)-.5*(np.outer(n,pb)+np.outer(pb,n)))
    lam,axes=eigh(K);assert lam.min()>0
    variances=1/lam;taus=c*tau0/lam
    energy=.5*c*delta**2-kappa*(n@pb)
    residual_free_energy=energy+kappa*(n@p)+.5*np.log(lam/c).sum()
    return dict(R=R,rotvec=rv,K=K,axes=axes,variances=variances,taus=taus,
                delta_deg=float(np.rad2deg(abs(delta))),rms_deg=float(np.rad2deg(np.sqrt(variances.sum()))),
                free_energy_residual=float(residual_free_energy),energy_kBT=float(energy),c=c)


def matched_h_response():
    """dH/d(field in MV/m), mean local orientation matched to frozen H.

    At local orientation Q, chi_lab[a,i,j]=Q[i,b]Q[j,c]chi[d,b,c]Q[a,d].
    This matters because the applied field is fixed in the protein frame.
    """
    _,Q,_,_,_,_=md.coefficient_series(2026091521,basis='svpd',external=True,tag='_dense')
    out={};raw={}
    for st in ('R','H','E'):
        z=dict(np.load(ROOT/f'data/field-couplings/{st}/response.npz'))
        from scipy.constants import physical_constants
        fieldau=physical_constants['atomic unit of electric field'][0]
        dv=z['dV_au_per_field_au']*electronic.EFG_AU_SI/fieldau*1e6
        da=z['dA_MHz_per_field_au']*2*np.pi*1e6/fieldau*1e6
        V=np.einsum('tnib,tnjc,ndbc,tnad->naij',Q,Q,dv,Q,optimize=True)/len(Q)
        A=np.einsum('tnib,tnjc,ndbc,tnad->naij',Q,Q,da,Q,optimize=True)/len(Q)
        raw[st]=dict(V=V,A=A)
        hn=np.zeros((3,9,9),complex)
        for a in range(3):
            for n in range(2):
                for i in range(3):
                    for j in range(3):
                        sym=(s.NI[n][i]@s.NI[n][j]+s.NI[n][j]@s.NI[n][i])/2
                        hn[a]+=md.PREF*V[n,a,i,j]*sym
        if st!='E':out[st]=hn
        else:
            out[st]=np.array([np.kron(np.eye(2),h) for h in hn])
            for a in range(3):
                for n in range(2):
                    for i in range(3):
                        for j in range(3):out[st][a]+=A[n,a,i,j]*np.kron(s.S[i],s.NI[n][j])
    out['RP']=np.array([np.einsum('anbm,cd->acnbdm',h.reshape(2,9,2,9),np.eye(2)).reshape(36,36) for h in out['E']])
    return out,raw


def coupled_generator(response,field_MVm):
    def build(f,angular,state,tau_s,field_T=50e-6):
        g=electric_generator(f,angular,state,tau_s,field_T)
        if field_MVm:
            for a,N in enumerate(angular['N']):g+=sp.kron(N,-1j*field_MVm*ad(response[state][a]),format='csc')
        return g
    return build


def lindblad_ops(f):
    out={}
    for st in ('H','R','E','RP'):
        K=f['archive']['K_'+('E' if st=='RP' else st)];v,u=eigh(K);ops=md.operators(st)
        out[st]=np.array([np.sqrt(x)*np.einsum('a,aij->ij',col,ops) for x,col in zip(v,u.T) if x>0])
    return out


def dissipator_sum(ops):
    d=ops.shape[-1]
    gain=np.einsum('aij,akl->ikjl',ops.conj(),ops,optimize=True).reshape(d*d,d*d)
    square=np.einsum('aji,ajk->ik',ops.conj(),ops,optimize=True)
    return gain-.5*(np.kron(np.eye(d),square)+np.kron(square.T,np.eye(d)))


class BoundFieldModel:
    def __init__(self,f,response,field_MVm,kappa,rms_deg,tau_w=1e-11,deform=True):
        self.f=f;self.response=response;self.field=field_MVm;self.kappa=kappa
        self.rms=rms_deg;self.tau=tau_w;self.deform=deform;self.jumps=lindblad_ops(f)
        self.records={};self.validity={}
    def at(self,st,n):
        f=self.f;w=well(n,self.rms,self.tau,self.kappa if self.deform else 0.)
        R0=w['R'];J=f['spin'][st]
        U0=expm(-1j*sum(w['rotvec'][a]*J[a] for a in range(3)))
        def value(R,U):
            ebody=self.field*(R.T@n)
            h=f['body'][st]+np.einsum('a,aij->ij',ebody,self.response[st])
            rotated=zgemm(1,zgemm(1,U,h),U.conj().T)
            # Enforce the exact Hermitian invariant before a small-angle
            # difference amplifies matrix-multiplication roundoff.
            return .5*(rotated+rotated.conj().T)
        meanH=np.zeros_like(f['body'][st]);rotated_ops=[];extra=np.zeros_like(f['D'][st])
        for a in range(3):
            direction=w['axes'][:,a];variance=w['variances'][a];tau=w['taus'][a]
            gen=sum(direction[j]*J[j] for j in range(3))
            eps=1e-5
            Up=expm(-1j*eps*gen);Um=Up.conj().T
            Rp=Rotation.from_rotvec(eps*direction).as_matrix();Rm=Rp.T
            v=(value(Rp@R0,zgemm(1,Up,U0))-value(Rm@R0,zgemm(1,Um,U0)))/(2*eps)
            extra+=2*variance*tau*dense_dissipator(v)
            angle=np.sqrt(3*variance)
            for sign in (-1,1):
                U=zgemm(1,expm(-1j*sign*angle*gen),U0);R=Rotation.from_rotvec(sign*angle*direction).as_matrix()@R0
                meanH+=value(R,U)/6
                rotated_ops.extend([zgemm(1,zgemm(1,U,L),U.conj().T)/np.sqrt(6) for L in self.jumps[st]])
        D=dissipator_sum(np.array(rotated_ops))+extra
        trace=s.trvec(len(meanH))
        current=dict(max_bandwidth_tau=float(np.ptp(eigh(meanH,eigvals_only=True))*max(w['taus'])),
                     max_trace_defect=float(np.linalg.norm(np.einsum('i,ij->j',trace,D))),
                     max_unital_defect=float(np.linalg.norm(np.einsum('ij,j->i',D,trace))))
        self.validity[st]={key:max(value,self.validity.get(st,{}).get(key,0)) for key,value in current.items()}
        if st=='H':
            self.records[tuple(n)]=dict(delta_deg=w['delta_deg'],rms_deg=w['rms_deg'],free_energy_residual=w['free_energy_residual'],covariance=np.einsum('ij,j,kj->ik',w['axes'],w['variances'],w['axes']).tolist(),taus_s=w['taus'].tolist())
        return dict(body=meanH,D=D)
