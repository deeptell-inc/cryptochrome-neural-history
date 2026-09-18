"""Conditional 14N dark-triplet chemistry feeding a physical product readout.

36-dimensional RP, 18-dimensional escaped flavin-electron/nuclear state,
9-dimensional closed-shell nuclei. Completed-cycle preparation is a theoretical
matched-chemistry protocol, not a measured clock or a physiological rate fit.
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from scipy.linalg import solve, eigh
from scipy import constants as c

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/dark-threshold'
TENSORS=ROOT/'data/dark-electronic-tensors'
MODELS=('full','population_z','population_HQ','scalar')


def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def vec(x):return np.asarray(x).reshape(-1,order='F')
def unvec(x,d):return np.asarray(x).reshape(d,d,order='F')
def trvec(d):return vec(np.eye(d))


def spins(j):
    m=np.arange(j,-j-1,-1);up=np.zeros((len(m),len(m)),complex)
    for col in range(1,len(m)):
        up[col-1,col]=np.sqrt(j*(j+1)-m[col]*(m[col]+1))
    return ((up+up.T)/2,(up-up.T)/(2j),np.diag(m))


S=spins(.5);I=spins(1)
NI=[[np.kron(x,np.eye(3)) for x in I],[np.kron(np.eye(3),x) for x in I]]
SINGLET=np.array([0,1,-1,0])/np.sqrt(2)
PS4=np.outer(SINGLET,SINGLET)
TRIPLET=(np.eye(4)-PS4)/3
X0=vec(np.eye(9)/9)
RESET=np.outer(X0,trvec(9))


def commutator_generator(h):
    eye=np.eye(len(h));return -1j*(np.kron(eye,h)-np.kron(h.T,eye))


def partial_electron(x):
    # One or both electrons are the leading tensor factor(s).
    d=len(x)//9
    return np.einsum('aiaj->ij',x.reshape(d,9,d,9))


def escape_state(x):
    return np.einsum('abcdbf->acdf',x.reshape(2,2,9,2,2,9)).reshape(18,18)


def linear_map(fun,din,dout):
    return np.column_stack([vec(fun(unvec(e,din))) for e in np.eye(din*din)])


TRACE_E=linear_map(partial_electron,18,9)
REPLACE_E=linear_map(lambda x:np.kron(np.eye(2)/2,partial_electron(x)),18,18)
REPLACE_N=linear_map(lambda x:np.kron(np.einsum('aibi->ab',x.reshape(2,9,2,9)),np.eye(9)/9),18,18)


def nuclear_hamiltonian(records,field_T,Q_barn):
    """Tabulated Q reference is declared, not an inferred relaxation time."""
    gamma=.403761*c.physical_constants['nuclear magneton'][0]/c.hbar
    h=-gamma*field_T*(NI[0][2]+NI[1][2]).astype(complex)
    pref=c.e*(Q_barn*1e-28)/(2*c.hbar)  # I=1: denominator 2 I(2I-1)
    for n,r in enumerate(records):
        v=np.asarray(r['EFG_V_m_minus2'])
        for a in range(3):
            for b in range(3):h+=pref*v[a,b]*(NI[n][a]@NI[n][b]+NI[n][b]@NI[n][a])/2
    return (h+h.conj().T)/2


def load_hamiltonians(basis,field_T=50e-6,Q_barn=.020443):
    names={'E':'E_neutral_SQ' if basis=='svpd' else 'E_neutral_SQ_pcJ2N_noDF',
           'R':'R_oxidized','H':'H_anionic_HQ'}
    rs={s:json.loads((TENSORS/name/'result.json').read_text()) for s,name in names.items()}
    hn={s:nuclear_hamiltonian(r['nuclei'],field_T,Q_barn) for s,r in rs.items()}
    af=np.array([r['HFC_rad_s'] for r in rs['E']['nuclei']])
    ge=abs(c.physical_constants['electron g factor'][0])
    we=ge*c.physical_constants['Bohr magneton'][0]*field_T/c.hbar
    he=np.kron(np.eye(2),hn['E'])+we*np.kron(S[2],np.eye(9))
    for n in range(2):
        for a in range(3):
            for b in range(3):he+=af[n,a,b]*np.kron(S[a],NI[n][b])
    # RP ordering is flavin electron, oxygen electron, N5, N10.
    h=np.kron(np.eye(4),hn['E'])+we*np.kron(np.kron(S[2],np.eye(2))+np.kron(np.eye(2),S[2]),np.eye(9))
    for n in range(2):
        for a in range(3):
            for b in range(3):h+=af[n,a,b]*np.kron(np.kron(S[a],np.eye(2)),NI[n][b])
    return h,he,hn,rs


def reaction_maps(h,gamma_oxygen,ks=1e7,ke=4e6):
    eye=np.eye(36);ps=np.kron(PS4,np.eye(9));loss=ks*ps+ke*eye
    L=commutator_generator(h)-.5*(np.kron(eye,loss)+np.kron(loss.T,eye))
    for s in S:
        op=np.kron(np.kron(np.eye(2),s),np.eye(9))
        L+=gamma_oxygen*(np.kron(op.conj(),op)-.5*np.kron(eye,op@op)-.5*np.kron((op@op).T,eye))
    injection=np.column_stack([vec(np.kron(TRIPLET,unvec(e,9))) for e in np.eye(81)])
    scale=max(ks,ke,gamma_oxygen,np.max(abs(h)))
    integ=solve(-L/scale,injection/scale,assume_a='gen')
    bra=np.kron(SINGLET[None,:],np.eye(9))
    p=np.column_stack([vec(ks*bra@unvec(x,36)@bra.conj().T) for x in integ.T])
    e=np.column_stack([vec(ke*escape_state(unvec(x,36))) for x in integ.T])
    residual=float(np.max(abs(trvec(9)@p+trvec(18)@e-trvec(9))))
    assert residual<1e-8
    return p,e,dict(trace_residual=residual,solve_relative_residual=float(np.linalg.norm((-L/scale)@integ-injection/scale)/np.linalg.norm(injection/scale)))


def exponential_wait(h,rate,gamma):
    """Exact exponential-residence average, global nuclear depolarization gamma."""
    if rate<=0 or gamma<0:raise ValueError('positive return and nonnegative damping required')
    w,u=eigh(h);ss=np.kron(u.conj(),u)
    factor=vec(rate/(rate+gamma+1j*(w[:,None]-w[None,:])))
    return (ss*factor)@ss.conj().T+gamma/(rate+gamma)*RESET


def cycle_map(p,e,he,hn,gamma_n,retention,a=10.,b=10.,cp=10.,oxygen_return=4.,gamma_flavin=1e8):
    if not 0<=retention<=1:raise ValueError('retention must be explicit probability')
    er=commutator_generator(he)+gamma_flavin*(REPLACE_E-np.eye(324))+gamma_n*(REPLACE_N-np.eye(324))
    scale=max(np.max(abs(er)),oxygen_return)
    system=(oxygen_return*np.eye(324)-er)/scale
    rhs=oxygen_return*e/scale
    # Substitute the analytically redundant population equation by exact trace
    # balance; avoids loss of mass from a GHz/second stiff linear solve.
    system[0]=trvec(18);rhs[0]=trvec(18)@e
    ew=solve(system,rhs)
    # Electron is retained during E waiting and traced only at E -> R oxidation.
    returned=exponential_wait(hn['R'],cp,gamma_n)@p+TRACE_E@ew
    returned=exponential_wait(hn['R'],a,gamma_n)@returned
    returned=(retention*np.eye(81)+(1-retention)*RESET)@returned
    m=exponential_wait(hn['H'],b,gamma_n)@returned
    return m


def population_projector(u):
    cols=np.column_stack([vec(np.outer(u[:,i],u[:,i].conj())) for i in range(9)])
    return cols@cols.conj().T,cols


def run():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'maps').mkdir(exist_ok=True)
    rows=[];quality=[];started=time.monotonic()
    for basis in ('svpd','pcJ2N'):
        h,he,hn,records=load_hamiltonians(basis)
        _,u=eigh(hn['H']);dhq,chq=population_projector(u);dz,cz=population_projector(np.eye(9))
        for ge in (0.,1e8):
            label=f'{basis}_ge{ge:g}'
            p,e,checks=reaction_maps(h,ge)
            y0=float((trvec(9)@p@X0).real)
            np.savez_compressed(OUT/'maps'/f'{label}.npz',product=p,escape=e,rp_hamiltonian_rad_s=h,escape_hamiltonian_rad_s=he,
                                H_R=hn['R'],H_H=hn['H'],H_E=hn['E'],HQ_basis=u)
            quality.append(dict(label=label,**checks))
            for gn in (0.,10.,1000.):
                for q in (0.,1.):
                    m=cycle_map(p,e,he,hn,gn,q)
                    assert np.max(abs(trvec(9)@m-trvec(9)))<2e-7
                    np.savez_compressed(OUT/'maps'/f'{label}_gn{gn:g}_q{q:g}.npz',cycle=m)
                    for model in MODELS:
                        state=X0.copy()
                        update={'full':m,'population_z':dz@m@dz,'population_HQ':dhq@m@dhq,'scalar':RESET@m@RESET}[model]
                        for n in range(21):
                            if n in (0,1,5,20):
                                rho=unvec(state,9);y=trvec(9)@p@state
                                assert abs(y.imag)<1e-8 and -1e-8<y.real<1+1e-8
                                rows.append(dict(basis=basis,gamma_oxygen_s=ge,gamma_nuclear_s=gn,retention=q,cycles=n,model=model,
                                    yield_P=float(y.real),yield_reference=y0,delta_yield=float(y.real-y0),
                                    burst_fraction=.05,product_fraction=.05*float(y.real),reference_product_fraction=.05*y0,
                                    trace_error=float(abs(np.trace(rho)-1)),minimum_eigenvalue=float(eigh((rho+rho.conj().T)/2,eigvals_only=True)[0]),
                                    acquisition='conditional calculation',native_calibrated=False))
                            state=update@state
            print(label,'unprepared P yield',y0,'elapsed_s',time.monotonic()-started,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'molecular.csv',index=False)
    save(OUT/'source_checks.json',quality)
    save(OUT/'configuration.json',dict(
        scope='Dark-triplet completed-cycle preparation; chemically matched HQ before a common probe. No native chronological preparation or macroscopic turnover calibration.',
        nuclei=['14N5','14N10'],nuclear_dimension=9,escaped_dimension=18,RP_dimension=36,
        field_T=50e-6,field_axis='PDB z, no orientation average',birth='unpolarized triplet PT/3',
        Q_barn=.020443,Q_status='declared software-table reference scenario; no new primary-source verification or native relaxation inference',
        Q_source='data/dark-electronic-tensors/software_sources/nucprop.py:157; upstream DOI 10.1016/j.adt.2015.12.002',
        kS_s=1e7,k_escape_s=4e6,exchange_rad_s=0.,electron_dipolar_rad_s=0.,
        omitted_couplings_status='J and electron dipolar terms set to zero as explicit uncalibrated scenario, not inferred absence',
        gamma_oxygen_s=[0,1e8],gamma_flavin_escape_s=1e8,gamma_nuclear_s=[0,10,1000],
        supply_s=10.,oxygen_birth_s=10.,product_recovery_s=10.,escape_oxidation_s=4.,direct_escape_reduction_s=0.,
        rates_status='all illustrative; return ratio .4 is a prior plant reference, not native mammalian calibration',
        retention_RH=[0,1],retention_status='global nuclear replacement channel at reduction; theoretical limiting cases',
        microscopic_adabatic_ratio=10/4e6,cycles=[0,1,5,20],probe_fraction=.05,
        product_readout='P branch only; not equated to total ROS. Equal-activity P+escape readout is a null control.',
        chemical_matching='All molecules returned to HQ before probe; previous active product removed/equalized as a theoretical control; prepare duration need not match across histories.',
        classical_comparison='Same single-RP quantum instrument and chemical waiting in all models. Population models retain 9 classical states between cycles; scalar retains none. These are not all-quantum-off models.',
        tensor_policy='Keep svpd and pcJ2N_noDF E tensors separate; R/H baseline tensors shared, cross-state basis sensitivity not fully explored',
        native_molecular_rates=None,native_neural_gain=None,total_energy_cost=None,elapsed_s=time.monotonic()-started))


if __name__=='__main__':run()
