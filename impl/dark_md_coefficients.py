"""Finite-window MD constraints; reduced states use explicitly counterfactual motion.

Rigid local tensor transport + direct environmental point-charge EFG.
No electronic polarization, intrinsic tensor changes, SOC, or native neural gain.
"""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from scipy import constants as c
from scipy.linalg import eigh,solve,expm,null_space
from scipy.spatial.transform import Rotation
from . import dark_threshold_source as s
from . import dark_motion_retention as old
from . import dark_threshold as neuron
from . import dark_finite_spike as finite
OUT=s.ROOT/'data/dark-md-coefficients'
SEEDS=(2026091521,2026091522,2026091523)
PREF=c.e*.020443e-28/(2*c.hbar)
B=np.array([np.diag([1.,-1,0])/np.sqrt(2),np.diag([-1.,-1,2])/np.sqrt(6),
            [[0,1/np.sqrt(2),0],[1/np.sqrt(2),0,0],[0,0,0]],
            [[0,0,1/np.sqrt(2)],[0,0,0],[1/np.sqrt(2),0,0]],
            [[0,0,0],[0,0,1/np.sqrt(2)],[0,1/np.sqrt(2),0]]])

def local_frame(x):
    a=x[...,1,:]-x[...,0,:];a=a/np.linalg.norm(a,axis=-1)[...,None]
    b=x[...,2,:]-x[...,0,:];z=np.cross(a,b);z=z/np.linalg.norm(z,axis=-1)[...,None]
    return np.stack([a,np.cross(z,a),z],axis=-1)

def coefficient_series(seed,basis='pcJ2N',external=False,tag=''):
    folder=OUT/(f'seed_{seed}'+tag)
    z=np.load(folder/'trajectory.npz');meta=json.loads((folder/'metadata.json').read_text())
    names=meta['ring_names'];x=z['ring_body_nm'];raw=s.load_hamiltonians(basis)[3]
    spec=json.loads((s.TENSORS/'R_oxidized/input.json').read_text());ref={a['label']:np.array(a['xyz_A']) for a in spec['atoms']}
    rotations=[]
    for labels,original in [(('N5','C4A','C5A'),('N5','C4X','C5X')),(('N10','C9A','C10'),('N10','C9A','C10'))]:
        f=local_frame(x[:,[names.index(a) for a in labels]]);f0=local_frame(np.array([ref[a] for a in original]));rotations.append(f@f0.T)
    Q=np.stack(rotations,axis=1);vs={};As=None
    for state in ('R','H','E'):
        v0=np.array([a['EFG_V_m_minus2'] for a in raw[state]['nuclei']]);v=Q@v0@Q.swapaxes(-1,-2)
        if external:v=v+z['external_EFG_V_m2']
        vs[state]=v
    A0=np.array([a['HFC_rad_s'] for a in raw['E']['nuclei']]);As=Q@A0@Q.swapaxes(-1,-2)
    coeff={k:(PREF*np.einsum('tnij,kij->tnk',v,B)).reshape(len(x),10) for k,v in vs.items()}
    ac=np.einsum('tnij,kij->tnk',As,B).reshape(len(x),10)
    coeff['E']=np.column_stack([coeff['E'],ac]);iso=np.trace(A0,axis1=-2,axis2=-1)/3
    return coeff,Q,vs,As,iso,z

def operators(state):
    nuc=[]
    for n in range(2):
        for b in B:
            nuc.append(sum(b[i,j]*(s.NI[n][i]@s.NI[n][j]+s.NI[n][j]@s.NI[n][i])/2 for i in range(3) for j in range(3)))
    if state in ('R','H'):return np.array(nuc)
    if state=='E':
        ops=[np.kron(np.eye(2),a) for a in nuc]
        for n in range(2):
            for b in B:ops.append(sum(b[i,j]*np.kron(s.S[i],s.NI[n][j]) for i in range(3) for j in range(3)))
    elif state=='RP':
        ops=[np.kron(np.eye(4),a) for a in nuc]
        for n in range(2):
            for b in B:ops.append(sum(b[i,j]*np.kron(np.kron(s.S[i],np.eye(2)),s.NI[n][j]) for i in range(3) for j in range(3)))
    return np.array(ops)

def fixed_hamiltonian(state,iso):
    gn=.403761*c.physical_constants['nuclear magneton'][0]/c.hbar
    hn=(-gn*50e-6*(s.NI[0][2]+s.NI[1][2])).astype(complex);we=abs(c.physical_constants['electron g factor'][0])*c.physical_constants['Bohr magneton'][0]*50e-6/c.hbar
    if state in ('R','H'):return hn.astype(complex)
    if state=='E':
        h=np.kron(np.eye(2),hn)+we*np.kron(s.S[2],np.eye(9))
        for n in range(2):
            for a in range(3):h+=iso[n]*np.kron(s.S[a],s.NI[n][a])
    else:
        h=np.kron(np.eye(4),hn)+we*np.kron(np.kron(s.S[2],np.eye(2))+np.kron(np.eye(2),s.S[2]),np.eye(9))
        for n in range(2):
            for a in range(3):h+=iso[n]*np.kron(np.kron(s.S[a],np.eye(2)),s.NI[n][a])
    return h.astype(complex)

def covariance_spectrum(x,dt,block):
    """PSD zero-frequency, finite rectangular integration-window estimate.

    E[(integral delta h dt)(integral delta h dt)^T]/T, all sliding windows.
    A Bartlett-filtered spectral estimate, NOT an infinite-time correlation fit.
    """
    y=x-x.mean(0);cs=np.vstack([np.zeros((1,x.shape[1])),np.cumsum(y,axis=0)])
    z=dt*(cs[block:]-cs[:-block]);K=z.T@z/(len(z)*block*dt)
    return (K+K.T)/2

def make_generator(state,x,iso,dt,block):
    ops=operators(state);K=covariance_spectrum(x,dt,block);ev,u=eigh(K)
    assert ev.min()>-max(abs(ev).max(),1)*1e-12
    H=fixed_hamiltonian(state,iso)+np.einsum('a,aij->ij',x.mean(0),ops)
    D=np.zeros((len(H)**2,len(H)**2),complex)
    for val,vec in zip(ev,u.T):
        if val>0:D+=val*old.dissipator(np.einsum('a,aij->ij',vec,ops))
    return s.commutator_generator(H)+D,H,D,K

def wait_map(L,rate):
    n=len(L);dim=round(np.sqrt(n));A=rate*np.eye(n)-L;rhs=rate*np.eye(n,dtype=complex)
    scale=np.max(abs(A));A=A/scale;rhs/=scale;A[0]=s.trvec(dim);rhs[0]=s.trvec(dim)
    return solve(A,rhs)

def reaction_maps(L):
    ps=np.kron(s.PS4,np.eye(9));loss=1e7*ps+4e6*np.eye(36)
    L=L-.5*(np.kron(np.eye(36),loss)+np.kron(loss.T,np.eye(36)))
    inj=np.column_stack([s.vec(np.kron(s.TRIPLET,s.unvec(v,9))) for v in np.eye(81)])
    scale=np.max(abs(L));integ=solve(-L/scale,inj/scale)
    bra=np.kron(s.SINGLET[None,:],np.eye(9))
    p=np.column_stack([s.vec(1e7*bra@s.unvec(v,36)@bra.conj().T) for v in integ.T])
    e=np.column_stack([s.vec(4e6*s.escape_state(s.unvec(v,36))) for v in integ.T])
    return p,e

def memory(L,H,state,t=.012):
    if state=='E':hn=np.einsum('aiaj->ij',H.reshape(2,9,2,9))/2
    else:hn=H
    _,u=eigh(hn);diag=np.column_stack([s.vec(np.outer(u[:,i],u[:,i].conj())) for i in range(9)])
    dp=diag@null_space(np.ones((1,9)));dc=np.column_stack([s.vec(np.outer(u[:,i],u[:,j].conj())) for i in range(9) for j in range(9) if i!=j])
    a=np.column_stack([dp,dc]);d=len(H)
    inj=a if d==9 else np.column_stack([s.vec(np.kron(np.eye(2)/2,s.unvec(v,9))) for v in a.T])
    # Remove the identity mode before propagation; avoids numerical trace drift.
    q=null_space(s.trvec(d)[None,:]);l=q.conj().T@L@q;W=q@expm(l*t)@(q.conj().T@inj)
    if d==18:W=s.TRACE_E@W
    return dict(population=float(np.linalg.norm(W[:,:8],'fro')/np.sqrt(8)),coherence=float(np.linalg.norm(W[:,8:],'fro')/np.sqrt(72)))

def correlation_diagnostics(seed):
    rows=[];acs=[];coeff,Q,vs,As,iso,z=coefficient_series(seed)
    dt=float(np.diff(z['time_ps'])[0])*1e-12
    for i,name in enumerate(('N5','N10')):
        mean=Q[:,i].mean(0);u,_,v=np.linalg.svd(mean);mean=u@v
        rv=Rotation.from_matrix(Q[:,i]@mean.T).as_rotvec();rms=float(np.sqrt(np.mean(np.sum(rv**2,axis=1)))*180/np.pi)
        for state in ('R','H','E'):
            values=[('EFG',vs[state][:,i])]
            if state=='E':values.append(('HFC',As[:,i]))
            for kind,x in values:
                x=x.reshape(len(x),-1);y=x-x.mean(0);v0=np.mean(np.sum(y*y,axis=1))
                corr=np.array([np.mean(np.sum(y[k:]*y[:len(y)-k],axis=1)) for k in range(251)])
                for lag,val in enumerate(corr):acs.append(dict(seed=seed,state=state,atom=name,kind=kind,lag_ps=lag*.02,C_over_C0=val/v0))
                for window in (1.,3.,5.):
                    K=covariance_spectrum(x,dt,round(window/.02))
                    rows.append(dict(seed=seed,state=state,atom=name,kind=kind,window_ps=window,
                        orientation_rms_deg=rms,tensor_rms=float(np.sqrt(v0)),mean_tensor_norm=float(np.linalg.norm(x.mean(0))),
                        finite_area_time_ps=float(np.trace(K)/(2*v0)*1e12),C5ps_over_C0=float(corr[-1]/v0),
                        first_half_variance=float(np.mean(np.sum((x[:len(x)//2]-x[:len(x)//2].mean(0))**2,axis=1))),
                        second_half_variance=float(np.mean(np.sum((x[len(x)//2:]-x[len(x)//2:].mean(0))**2,axis=1)))))
    return rows,acs

def run(seed,tag=''):
    started=time.monotonic();dest=OUT/(f'seed_{seed}'+tag);(dest/'maps').mkdir(exist_ok=True)
    rows=[];mem=[];elec=[];spectra=[]
    if not tag:
        cor,ac=correlation_diagnostics(seed)
        pd.DataFrame(cor).to_csv(dest/'correlation_summary.csv',index=False);pd.DataFrame(ac).to_csv(dest/'autocorrelations.csv',index=False)
    for external in (False,True):
        coeff,Q,vs,As,iso,z=coefficient_series(seed,external=external,tag=tag);dt=float(np.diff(z['time_ps'])[0])*1e-12
        np.savez_compressed(dest/f'tensors_ext{int(external)}.npz',**{f'EFG_{k}':v for k,v in vs.items()},HFC_E=As,rotation_local=Q,**{f'coeff_{k}':v for k,v in coeff.items()})
        for window in ((1.,3.) if tag else (1.,3.,5.)):
            block=round(window*1e-12/dt);g={};hs={};ds={};ks={}
            for state in ('R','H','E','RP'):
                g[state],hs[state],ds[state],ks[state]=make_generator(state,coeff['E' if state=='RP' else state],iso,dt,block)
            labelbase=f'ext{int(external)}_w{window:g}'
            np.savez_compressed(dest/'maps'/f'{labelbase}_generators.npz',**{f'L_{k}':v for k,v in g.items()},**{f'H_{k}':v for k,v in hs.items()},**{f'K_{k}':v for k,v in ks.items()})
            for state in ('R','H','E'):
                for t in ((1e-6,1e-4,.012) if state=='E' else (.012,)):
                    mem.append(dict(seed=seed,external=external,window_ps=window,state=state,time_s=t,**memory(g[state],hs[state],state,t)))
                x=coeff[state];vv=np.mean(np.sum((x-x.mean(0))**2,axis=1));spectra.append(dict(seed=seed,external=external,window_ps=window,state=state,
                    sqrt_coefficient_variance_rad_s=float(np.sqrt(vv)),finite_area_time_ps=float(np.trace(ks[state])/(2*vv)*1e12),
                    mean_H_span_rad_s=float(np.ptp(eigh(hs[state],eigvals_only=True))),white_flatness_parameter=float(np.ptp(eigh(hs[state],eigvals_only=True))*window*1e-12)))
            for axis in range(3):
                op=np.kron(s.S[axis],np.eye(9));v=s.vec(op)
                rate=float(-(v.conj()@ds['E']@v).real/(v.conj()@v).real)
                elec.append(dict(seed=seed,external=external,window_ps=window,axis=axis,motion_dissipative_electron_rate_s=rate,
                    scope='initial decay projection from fluctuating N5/N10 HFC only; not total electron T1/T2 or scalar replacement rate'))
            wr=wait_map(g['R'],10);wh=wait_map(g['H'],10);_,u=eigh(hs['H']);dephase,_=s.population_projector(u)
            for ge in (0.,1e8):
                L=g['RP'].copy()
                for sp in s.S:L+=ge*old.dissipator(np.kron(np.kron(np.eye(2),sp),np.eye(9)))
                p,e=reaction_maps(L);yref=float((s.trvec(9)@p@s.X0).real)
                for gamma_extra in (0.,1e8):
                    we=wait_map(g['E']+gamma_extra*(s.REPLACE_E-np.eye(324)),4.)
                    cyc=wh@wr@(wr@p+s.TRACE_E@we@e)
                    label=f'{labelbase}_go{ge:g}_extra{gamma_extra:g}'
                    np.savez_compressed(dest/'maps'/f'{label}.npz',cycle=cyc,product=p,escape=e,HQ_basis=u)
                    flow=float(np.max(abs(s.trvec(9)@p+s.trvec(18)@e-s.trvec(9))))
                    mass=float(np.max(abs(s.trvec(9)@cyc-s.trvec(9))))
                    assert max(flow,mass)<1e-7
                    baseline=finite.first_passage(.05*yref,.05*yref,.012,'tonic_excess',n=1200)['firing_probability']
                    for model,update in [('full',cyc),('population_HQ',dephase@cyc@dephase),('scalar',s.RESET@cyc@s.RESET)]:
                        x=np.linalg.matrix_power(update,20)@s.X0;y=float((s.trvec(9)@p@x).real)
                        ns=neuron.spike_metrics(.05*y,.05*yref,100.);fp=finite.first_passage(.05*y,.05*yref,.012,'tonic_excess',n=1200)
                        rows.append(dict(seed=seed,external=external,window_ps=window,gamma_oxygen_s=ge,gamma_extra_escape_s=gamma_extra,model=model,
                            yield_reference=yref,yield_P=y,delta_yield=y-yref,delta_spike_12ms=fp['firing_probability']-baseline,
                            delta_current_pA=ns['delta_current_pA'],flow_error=flow,mass_error=mass,
                            trace_error=float(abs(s.trvec(9)@x-1)),minimum_state_eigenvalue=float(eigh(s.unvec(x,9),eigvals_only=True)[0])))
            print(seed,labelbase,'elapsed',time.monotonic()-started,flush=True)
            pd.DataFrame(rows).to_csv(dest/'products_spikes.csv',index=False);pd.DataFrame(mem).to_csv(dest/'memory.csv',index=False)
            pd.DataFrame(elec).to_csv(dest/'electron_rates.csv',index=False);pd.DataFrame(spectra).to_csv(dest/'spectra.csv',index=False)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--tag',default='');a=p.parse_args();run(a.seed,a.tag)
