"""Write--retain--read audit of the existing two-14N dark CRY instrument.

Pauli modes are conditional generator modes, not measured biological T1 values.
A boundary-population model still uses quantum reaction and recovery maps.
"""
from pathlib import Path
import importlib.util,json
import numpy as np
from scipy.linalg import eigh,expm,null_space
from . import dark_threshold_source as s
from . import dark_md_coefficients as md
from . import dark_motion_retention as motion
ROOT=s.ROOT;OUT=ROOT/'data/population-memory'


def build_case(label='upcj2N__full',window_ps=1.):
    path=ROOT/'investigations/2026-09-16-dark-basis-resolution/propagate.py'
    spec=importlib.util.spec_from_file_location('frozen_basis_propagation',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=json.loads((ROOT/'data/dark-basis-resolution'/label/'result.json').read_text())
    coeff,iso,dt=module.coefficients(result)
    g={};h={};d={};k={}
    for state in ('R','H','E','RP'):
        g[state],h[state],d[state],k[state]=md.make_generator(state,coeff['E' if state=='RP' else state],iso,dt,round(window_ps*1e-12/dt))
    maps={}
    for gamma in (0.,1e8):
        L=g['RP'].copy()
        for spin in s.S:L+=gamma*motion.dissipator(np.kron(np.kron(np.eye(2),spin),np.eye(9)))
        p,e=md.reaction_maps(L);maps[gamma]=(p,e)
    data={**{f'L_{a}':g[a] for a in ('R','H','E')},**{f'H_{a}':h[a] for a in ('R','H','E')},
          **{f'D_{a}':d[a] for a in ('R','H','E')},**{f'K_{a}':k[a] for a in ('R','H','E')},
          **{f'P_{x:g}':maps[x][0] for x in maps},**{f'Escape_{x:g}':maps[x][1] for x in maps}}
    return data


def population_geometry(H,D):
    energies,u=eigh(H);projector,C=s.population_projector(u)
    Q=np.column_stack([s.vec(np.outer(u[:,i],u[:,j].conj())) for i in range(9) for j in range(9) if i!=j])
    raw=C.conj().T@D@C
    assert np.max(abs(raw.imag))<1e-8
    G=raw.real
    # Hermitian noise operators imply a symmetric, uniform-stationary Pauli generator.
    assert np.max(abs(G-G.T))<1e-8 and np.max(abs(G.sum(0)))<1e-8
    G=(G+G.T)/2
    off=G.copy();np.fill_diagonal(off,0)
    assert off.min()>-1e-10
    off=np.maximum(off,0);G=off.copy();np.fill_diagonal(G,-off.sum(0))
    T=null_space(np.ones((1,9)))
    rates,V=eigh(-T.T@G@T);V=T@V
    for j in range(8):
        if V[np.argmax(abs(V[:,j])),j]<0:V[:,j]*=-1
    return dict(energies=energies,U=u,C=C,Q=Q,P=projector,G=G,rates=rates,V=V,
                roundoff_adjustment=float(np.linalg.norm(G-raw.real)))


def character(operator):
    x=np.asarray(operator).reshape(3,3,3,3)
    n5=np.einsum('abcb->ac',x);n10=np.einsum('abad->bd',x)
    local5=np.kron(n5,np.eye(3)/3);local10=np.kron(np.eye(3)/3,n10)
    corr=operator-local5-local10
    norm=np.linalg.norm(operator)**2
    weights=[np.linalg.norm(z)**2/norm for z in (local5,local10,corr)]
    magnetic=[]
    for marginal in (n5,n10):
        magnetic.append(sum(abs(np.trace(sp.conj().T@marginal))**2/np.trace(sp.conj().T@sp).real for sp in s.I)/(3*norm))
    return dict(N5_local_fraction=float(weights[0]),N10_local_fraction=float(weights[1]),
                correlation_fraction=float(weights[2]),magnetization_fraction=float(sum(magnetic)),
                local_rank2_fraction=float(sum(weights[:2])-sum(magnetic)))


def cycle(data,gamma=0.,a=10.,b=10.,cp=10.,ce=4.,retention=1.,noise_scale=1.,phase_rate=0.,reset_at='reduction',spinblind=False,
          h_noise_scale=None,r_noise_scale=None,escape_noise_scale=1.):
    """Full within-cycle evolution; modified rates are explicit interventions.

    noise_scale changes the entire closed-shell dissipator (P and C), not only P.
    E and RP generators are held fixed. Phase_rate dephases HQ without directly
    changing its populations. Reset is an extra ideal nuclear replacement map.
    """
    P=data[f'P_{gamma:g}'];E=data[f'Escape_{gamma:g}']
    if spinblind:
        y0=float((s.trvec(9)@P@s.X0).real)
        P=y0*np.eye(81)
        E=(1-y0)*np.column_stack([s.vec(np.kron(np.eye(2)/2,s.unvec(x,9))) for x in np.eye(81)])
    geo=population_geometry(data['H_H'],data['D_H']);DP=geo['P']
    hs=noise_scale if h_noise_scale is None else h_noise_scale
    rs=noise_scale if r_noise_scale is None else r_noise_scale
    LH=s.commutator_generator(data['H_H'])+hs*data['D_H']+phase_rate*(DP-np.eye(81))
    LR=s.commutator_generator(data['H_R'])+rs*data['D_R']
    LE=s.commutator_generator(data['H_E'])+escape_noise_scale*data['D_E']
    WH=md.wait_map(LH,b);WA=md.wait_map(LR,a);WP=md.wait_map(LR,cp);WE=md.wait_map(LE,ce)
    reset=retention*np.eye(81)+(1-retention)*s.RESET
    if reset_at=='reaction':
        replace_E=retention*np.eye(324)+(1-retention)*s.REPLACE_N
        P=reset@P;E=replace_E@E
    returned=WA@(WP@P+s.TRACE_E@WE@E)
    if reset_at=='reduction':returned=reset@returned
    if reset_at not in ('reaction','reduction'):raise ValueError(reset_at)
    return WH@returned,dict(before_H=returned,Hwait=WH,Rwait=WA,Pwait=WP,Ewait=WE,LH=LH,LR=LR)


def compare_cycle(M,P,geo,n=20):
    DP=geo['P'];r=s.trvec(9)@P;y0=float((r@s.X0).real)
    rows=[];states={}
    for name,update in [('full',M),('population_HQ',DP@M@DP),('reset',s.RESET@M@s.RESET)]:
        x=np.linalg.matrix_power(update,n)@s.X0;rho=s.unvec(x,9)
        rows.append(dict(model=name,cycles=n,yield_P=float((r@x).real),delta_yield=float((r@x).real-y0),
                         trace_error=float(abs(s.trvec(9)@x-1)),minimum_eigenvalue=float(eigh((rho+rho.conj().T)/2,eigvals_only=True)[0]),
                         boundary_coherence_norm=float(np.linalg.norm((np.eye(81)-DP)@x))))
        states[name]=x
    return rows,states


def wait_spectrum(L):
    """Use traceless space so a numerically drifting identity never propagates."""
    T=null_space(s.trvec(9)[None,:]);a=T.conj().T@L@T
    from scipy.linalg import eig
    values,vectors=eig(a)
    return T,values,vectors,np.linalg.inv(vectors)


def propagate_traceless(spectrum,x,t):
    T,values,vectors,inverse=spectrum
    return T@(vectors@(np.exp(values*t)*(inverse@(T.conj().T@x))))


def mode_table(geo,P,state):
    C,V=geo['C'],geo['V'];r=(s.trvec(9)@P@C).real
    w=(C.conj().T@(state-s.X0)).real;write=V.T@w;read=r@V
    rows=[]
    for j in range(8):
        rows.append(dict(mode=j+1,decay_rate_s=float(geo['rates'][j]),
                         time_constant_s=float(1/geo['rates'][j]) if geo['rates'][j]>1e-14 else None,
                         write_amplitude=float(write[j]),read_sensitivity=float(read[j]),
                         contribution_at_zero=float(write[j]*read[j]),
                         **character(s.unvec(C@V[:,j],9))))
    return rows


def markov_wait(G,rate):return np.linalg.solve(rate*np.eye(9)-G,rate*np.eye(9))


def unital_wait(L,rate):
    """Independent resolvent with the exact identity invariant removed.

    Only applicable to the unital, trace-preserving waiting generators here.
    Avoids treating stiff-solve drift of a uniform state as physical writing.
    """
    dim=round(np.sqrt(len(L)));eye=s.trvec(dim)
    assert max(np.linalg.norm(L@eye),np.linalg.norm(eye@L))<1e-6
    T=null_space(eye[None,:]);reduced=T.conj().T@L@T
    return np.outer(eye,eye)/dim+T@np.linalg.solve(rate*np.eye(len(reduced))-reduced,rate*T.conj().T)


def no_bath_case(data):
    """Counterfactual: all external dissipators off, retaining reactive losses.

    RP/E coherent dynamics, spin-selection, electron trace and chemical
    residence averaging remain; this does NOT switch off quantum chemistry.
    """
    out=dict(data)
    for state in ('R','H','E'):
        out['D_'+state]=np.zeros_like(data['D_'+state])
        out['L_'+state]=s.commutator_generator(data['H_'+state])
    he=data['H_E'];ze=he.reshape(2,9,2,9).trace(axis1=1,axis2=3)/9
    hrp=np.einsum('anbm,op->aonbpm',he.reshape(2,9,2,9),np.eye(2)).reshape(36,36)
    hrp+=np.kron(np.kron(np.eye(2),ze),np.eye(9))
    out['P_0'],out['Escape_0']=md.reaction_maps(s.commutator_generator(hrp))
    return out


def population_jump_lift(geo):
    """Diagnostic CP lift of a Pauli generator; not identical to original D.

    Each transition |i><j| is its own Lindblad jump. This choice changes coherence
    damping (including degenerate transitions), so is a sensitivity construction.
    """
    D=np.zeros((81,81),complex);u=geo['U'];G=geo['G']
    for i in range(9):
        for j in range(9):
            if i!=j and G[i,j]>0:
                D+=G[i,j]*motion.dissipator(np.outer(u[:,i],u[:,j].conj()))
    return D
