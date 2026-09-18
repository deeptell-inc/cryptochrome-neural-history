"""Exact two-orientation stochastic-Liouville retention, conditionally parameterized.

State-resolved static 14N tensors are rotated together. The telegraph coordinate
has C(t)=exp(-|t|/tau); its amplitude/time are scenarios, not native MD estimates.
Orientation remains a classical state through reaction, escape and recovery.
"""
from functools import lru_cache
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from scipy.linalg import solve,eigh,eig,null_space,expm
from scipy.spatial.transform import Rotation
from . import dark_threshold_source as s
from . import dark_threshold as neuron
from . import dark_finite_spike as finite

OUT=s.ROOT/'data/dark-motion-retention'


def dissipator(op):
    eye=np.eye(len(op));a=op.conj().T@op
    return np.kron(op.conj(),op)-.5*(np.kron(eye,a)+np.kron(a.T,eye))


@lru_cache(None)
def state_pair(basis,angle_deg,axis):
    h,he,hn,records=s.load_hamiltonians(basis)
    pair=[];tensor_rows=[]
    for sign in (1,-1):
        R=Rotation.from_rotvec(np.eye(3)[axis]*np.deg2rad(angle_deg)*sign).as_matrix()
        nr={};rr={}
        for state,recs in records.items():
            nuclei=[]
            for item in recs['nuclei']:
                a=dict(item);a['EFG_V_m_minus2']=(R@np.asarray(item['EFG_V_m_minus2'])@R.T).tolist()
                if item['HFC_rad_s'] is not None:a['HFC_rad_s']=(R@np.asarray(item['HFC_rad_s'])@R.T).tolist()
                nuclei.append(a)
            rr[state]=nuclei;nr[state]=s.nuclear_hamiltonian(nuclei,50e-6,.020443)
        # Derive the shared electron Zeeman term from the original loaded model.
        original_hfc=np.array([x['HFC_rad_s'] for x in records['E']['nuclei']])
        e_zeeman=he-np.kron(np.eye(2),hn['E'])
        for n in range(2):
            for a in range(3):
                for b in range(3):e_zeeman-=original_hfc[n,a,b]*np.kron(s.S[a],s.NI[n][b])
        new_he=np.kron(np.eye(2),nr['E'])+e_zeeman
        new_rp=np.kron(np.eye(4),nr['E'])
        # The same fixed laboratory field acts on both electrons.
        electron_z=e_zeeman.reshape(2,9,2,9).trace(axis1=1,axis2=3)/9
        new_rp+=np.kron(np.kron(electron_z,np.eye(2))+np.kron(np.eye(2),electron_z),np.eye(9))
        for n,item in enumerate(rr['E']):
            A=np.asarray(item['HFC_rad_s'])
            for a in range(3):
                for b in range(3):
                    new_he+=A[a,b]*np.kron(s.S[a],s.NI[n][b])
                    new_rp+=A[a,b]*np.kron(np.kron(s.S[a],np.eye(2)),s.NI[n][b])
        pair.append(dict(R=nr['R'],H=nr['H'],E=new_he,RP=new_rp,En=nr['E']))
        tensor_rows.append(rr)
    return pair,tensor_rows


def generators(pair,ge):
    out=[]
    for h in pair:
        g={k:s.commutator_generator(h[k]) for k in ('R','H','E','RP')}
        g['E']+=1e8*(s.REPLACE_E-np.eye(324))
        ps=np.kron(s.PS4,np.eye(9));loss=1e7*ps+4e6*np.eye(36)
        g['RP']-=.5*(np.kron(np.eye(36),loss)+np.kron(loss.T,np.eye(36)))
        for spin in s.S:
            op=np.kron(np.kron(np.eye(2),spin),np.eye(9));g['RP']+=ge*dissipator(op)
        out.append(g)
    return out


def resolvent(L0,L1,tau,rate,rhs0,rhs1,trace_dim=None):
    """Solve in sum/difference coordinates, eliminating the fast telegraph block.

    For waits, rhs=rate*rho. For RP integration rate=0 and L includes loss.
    tau=0 is the exact infinite-switching mean-Hamiltonian limit.
    """
    n=len(L0);A=rate*np.eye(n)-(L0+L1)/2;B=-(L0-L1)/2
    plus=rhs0+rhs1;minus=rhs0-rhs1
    if tau==0:
        eff=A.copy();rhs=plus.copy();Dinv=None
    else:
        D=A+np.eye(n)/tau
        db=solve(D,np.column_stack([B,minus]),assume_a='gen',check_finite=False)
        DB,DM=db[:,:n],db[:,n:]
        eff=A-B@DB;rhs=plus-B@DM
    if trace_dim is not None:
        assert rate>0
        eff[0]=s.trvec(trace_dim);rhs[0]=(s.trvec(trace_dim)@plus)/rate
    scale=max(float(np.max(abs(eff))),1.)
    if trace_dim is not None:
        # Preserve the trace equation's relative conditioning after scaling.
        eff[0]*=scale;rhs[0]*=scale
    total=solve(eff/scale,rhs/scale,assume_a='gen',check_finite=False)
    diff=np.zeros_like(total) if tau==0 else DM-DB@total
    return (total+diff)/2,(total-diff)/2


def wait_map(gens,state,tau,rate):
    n=len(gens[0][state]);dim=round(np.sqrt(n));zero=np.zeros((n,n));ident=rate*np.eye(n)
    r0=np.column_stack([ident,zero]);r1=np.column_stack([zero,ident])
    x,y=resolvent(gens[0][state],gens[1][state],tau,rate,r0,r1,dim)
    return np.vstack([x,y])


def reaction_maps(gens,tau):
    inj=np.column_stack([s.vec(np.kron(s.TRIPLET,s.unvec(e,9))) for e in np.eye(81)])
    zero=np.zeros_like(inj)
    r0=np.column_stack([inj,zero]);r1=np.column_stack([zero,inj])
    x,y=resolvent(gens[0]['RP'],gens[1]['RP'],tau,0.,r0,r1)
    bra=np.kron(s.SINGLET[None,:],np.eye(9))
    p=[];e=[]
    for z in (x,y):
        p.append(np.column_stack([s.vec(1e7*bra@s.unvec(v,36)@bra.conj().T) for v in z.T]))
        e.append(np.column_stack([s.vec(4e6*s.escape_state(s.unvec(v,36))) for v in z.T]))
    return np.vstack(p),np.vstack(e)


def cycle_maps(gens,tau,p,e):
    wp=wait_map(gens,'R',tau,10.)
    wr=wp;wh=wait_map(gens,'H',tau,10.)
    we=wait_map(gens,'E',tau,4.)
    trace_e=np.kron(np.eye(2),s.TRACE_E)
    cyc=wh@wr@(wp@p+trace_e@we@e)
    return cyc,dict(R=wr,H=wh,E=we)


def memory_blocks(gens,pair,state,tau,times=(.012,.1)):
    """Fixed-time mean-square mode survival, in the mean nuclear energy basis.

    These are averages over all orthonormal traceless population/coherence modes,
    not fitted T1/T2 and not an exponential residence average.
    """
    dim=9 if state!='E' else 18;n=dim*dim
    q=null_space(s.trvec(dim)[None,:])
    avg=q.conj().T@((gens[0][state]+gens[1][state])/2)@q
    dif=q.conj().T@((gens[0][state]-gens[1][state])/2)@q
    if tau==0:L=avg
    else:L=np.block([[avg,dif],[dif,avg-np.eye(n-1)/tau]])
    scale=max(np.max(abs(L)),1.)
    lam,V=eig(L/scale);lam*=scale
    hn=(pair[0]['En' if state=='E' else state]+pair[1]['En' if state=='E' else state])/2
    _,u=eigh(hn);columns=np.column_stack([s.vec(np.outer(u[:,i],u[:,i].conj())) for i in range(9)])
    bp=columns@null_space(np.ones((1,9)))
    bc=np.column_stack([s.vec(np.outer(u[:,i],u[:,j].conj())) for i in range(9) for j in range(9) if i!=j])
    basis=np.column_stack([bp,bc])
    # Local N5/N10 operators embedded with a normalized spectator identity.
    hn4=hn.reshape(3,3,3,3)
    local_h=[np.einsum('abcb->ac',hn4)/3,np.einsum('abad->bd',hn4)/3]
    local={}
    for site,name in enumerate(('N5','N10')):
        _,ul=eigh(local_h[site])
        dp=np.column_stack([s.vec(np.outer(ul[:,i],ul[:,i].conj())) for i in range(3)])@null_space(np.ones((1,3)))
        dc=np.column_stack([s.vec(np.outer(ul[:,i],ul[:,j].conj())) for i in range(3) for j in range(3) if i!=j])
        for kind,mat in [('population',dp),('coherence',dc)]:
            embedded=np.column_stack([s.vec(np.kron(s.unvec(v,3),np.eye(3)/np.sqrt(3)) if site==0 else np.kron(np.eye(3)/np.sqrt(3),s.unvec(v,3))) for v in mat.T])
            local[(name,kind)]=basis.conj().T@embedded
    inj=basis if state!='E' else np.column_stack([s.vec(np.kron(np.eye(2)/2,s.unvec(x,9))) for x in basis.T])
    rhs=q.conj().T@inj
    if tau!=0:rhs=np.vstack([rhs,np.zeros_like(rhs)])
    coeff=solve(V,rhs)
    rows=[]
    residual=np.linalg.norm((L/scale)@V-V*(lam/scale))/np.linalg.norm(V)
    for time_s in times:
        propagated=V@(np.exp(lam*time_s)[:,None]*coeff)
        total=q@propagated[:n-1]
        if state=='E':total=s.TRACE_E@total
        ptotal=np.linalg.norm(total[:,:8],'fro')/np.sqrt(8)
        ctotal=np.linalg.norm(total[:,8:],'fro')/np.sqrt(72)
        pp=np.linalg.norm(bp.conj().T@total[:,:8],'fro')/np.sqrt(8)
        cc=np.linalg.norm(bc.conj().T@total[:,8:],'fro')/np.sqrt(72)
        local_values={f'{name}_{kind}_input_retention':float(np.linalg.norm(total@c,'fro')/np.sqrt(c.shape[1])) for (name,kind),c in local.items()}
        rows.append(dict(state=state,time_s=time_s,population_input_total_retention=float(ptotal),coherence_input_total_retention=float(ctotal),
            population_to_population=float(pp),coherence_to_coherence=float(cc),
            largest_real_eigenvalue_s=float(lam.real.max()),eigen_relative_residual=float(residual),**local_values))
    return rows


def run():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'maps').mkdir(exist_ok=True)
    specs=[]
    for basis in ('svpd','pcJ2N'):
        for ge in (0.,1e8):
            specs.append((basis,0.,0,1e-10,ge))
            for angle in (1.,5.):
                for tau in (0.,1e-12,1e-10,1e-8):specs.append((basis,angle,0,tau,ge))
    for axis in (1,2):
        for ge in (0.,1e8):specs.append(('pcJ2N',5.,axis,1e-10,ge))
    rows=[];memory=[];corr=[];started=time.monotonic()
    seen_memory=set()
    for basis,angle,axis,tau,ge in specs:
        label=f'{basis}_a{angle:g}_axis{axis}_tau{tau:g}_ge{ge:g}'
        pair,tensors=state_pair(basis,angle,axis);g=generators(pair,ge)
        tensor_file=OUT/'tensor_pairs'/f'{basis}_a{angle:g}_axis{axis}.json'
        if not tensor_file.exists():
            s.save(tensor_file,dict(basis=basis,angle_deg=angle,axis=axis,states=tensors,
                status='static electronic tensors rotated into two assumed orientations; not sampled QM dynamics'))
        p,e=reaction_maps(g,tau);cyc,waits=cycle_maps(g,tau,p,e)
        trace=np.tile(s.trvec(9),2);trace18=np.tile(s.trvec(18),2)
        flowerr=float(np.max(abs(trace@p+trace18@e-trace)))
        masserr=float(np.max(abs(trace@cyc-trace)))
        assert flowerr<1e-7 and masserr<1e-7,(label,flowerr,masserr)
        initial=np.tile(s.X0/2,2)
        _,u=eigh((pair[0]['H']+pair[1]['H'])/2);proj,_=s.population_projector(u)
        dephase=np.kron(np.eye(2),proj);reset=np.kron(np.eye(2),s.RESET)
        yref=float((trace@p@initial).real)
        for model,up in [('full',cyc),('population_HQ',dephase@cyc@dephase),('scalar',reset@cyc@reset)]:
            state=initial.copy()
            for i in range(20):state=up@state
            y=float((trace@p@state).real);ref=yref
            eigmin=min(eigh(s.unvec(state[j*81:(j+1)*81],9),eigvals_only=True)[0] for j in range(2))
            if model=='scalar':assert abs(y-yref)<1e-7
            ns=neuron.spike_metrics(.05*y,.05*ref,100.);base=neuron.spike_metrics(.05*ref,.05*ref,100.)
            fp=finite.first_passage(.05*y,.05*ref,.012,'tonic_excess',n=1200)
            fb=finite.first_passage(.05*ref,.05*ref,.012,'tonic_excess',n=1200)
            rows.append(dict(label=label,basis=basis,angle_deg=angle,axis=axis,tau_c_s=tau,gamma_oxygen_s=ge,model=model,
                yield_reference=yref,yield_P=y,delta_yield=y-yref,delta_current_pA=ns['delta_current_pA'],
                delta_spike_probability_held=ns['firing_probability']-base['firing_probability'],
                delta_spike_probability_tau12ms=fp['firing_probability']-fb['firing_probability'],
                minimum_sector_eigenvalue=float(eigmin),trace_error=float(abs(trace@state-1)),reaction_flow_error=flowerr,cycle_mass_error=masserr))
        np.savez_compressed(OUT/'maps'/f'{label}.npz',product=p,escape=e,cycle=cyc,initial=initial,HQ_basis=u,**{f'wait_{k}':v for k,v in waits.items()})
        key=(basis,angle,axis,tau)
        if key not in seen_memory:
            for state_name in ('R','H','E'):
                mm=memory_blocks(g,pair,state_name,tau,times=(1e-6,1e-5,1e-4,.012,.1) if state_name=='E' else (.012,.1))
                memory.extend([dict(basis=basis,angle_deg=angle,axis=axis,tau_c_s=tau,**v) for v in mm])
            for state_name in ('R','H','E'):
                for ni,atom in enumerate(('N5','N10')):
                    for quantity in ('EFG_V_m_minus2','HFC_rad_s'):
                        a=tensors[0][state_name][ni][quantity];b=tensors[1][state_name][ni][quantity]
                        if a is None:continue
                        mean=(np.array(a)+b)/2;diff=(np.array(a)-b)/2
                        corr.append(dict(basis=basis,angle_deg=angle,axis=axis,tau_c_s=tau,state=state_name,atom=atom,quantity=quantity,
                            mean_frobenius=float(np.linalg.norm(mean)),fluctuation_rms_frobenius=float(np.linalg.norm(diff)),
                            integrated_tensor_covariance_trace=float(np.sum(diff*diff)*tau),
                            correlation_model='common telegraph: C_AB(t)=DeltaA tensor-product DeltaB exp(-abs(t)/tau); tau=0 mean-H limit'))
            seen_memory.add(key)
        print(label,'flow',flowerr,'cycle',masserr,'elapsed',time.monotonic()-started,flush=True)
        # Checkpoint tables so interrupted runs remain inspectable.
        pd.DataFrame(rows).to_csv(OUT/'products_spikes.csv',index=False)
        pd.DataFrame(memory).to_csv(OUT/'memory_modes.csv',index=False)
        pd.DataFrame(corr).to_csv(OUT/'tensor_correlations.csv',index=False)
    s.save(OUT/'configuration.json',dict(date='2026-09-16',scenarios=len(specs),basis=['svpd','pcJ2N'],angle_deg=[0,1,5],
        tau_c_s=[0,1e-12,1e-10,1e-8],tau_zero='infinite-switching mean-Hamiltonian limit, not zero motion',
        axes='x main, y/z sensitivity for pcJ2N 5deg 100ps',motion='two orientations, symmetric Markov switching at 1/(2*tau)',
        native_fluctuation_series_available=False,amplitude_and_correlation_time_status='assumed discrete scenarios, not measured priors',
        rotating_quantities='state-dependent N5/N10 EFG and anisotropic HFC; isotropic HFC preserved; laboratory Zeeman fixed',
        global_nuclear_reset_gamma_s=0,retention_RH=1,gamma_flavin_escape_s=1e8,product_recovery_s=10,supply_s=10,birth_s=10,escape_return_s=4,
        model_comparison='all share same single-RP instrument; full/HQ-population/scalar differ only between completed cycles; telegraph label retained',
        memory_metrics='Frobenius mode retention of traceless populations/coherences in mean nuclear energy basis at fixed .012/.1 seconds; no single T1/T2 fit',
        protected_state_warning='one-axis motion can create symmetries absent from general 3D fluctuations; no inference of native DD',
        neural_scenario='unchanged G100pA, C20pF, I04pA, noise20mV/sqrt(s), threshold10mV, 50ms first passage; held and 12ms excess decay',
        biological_replicates=0,native_relaxation=None,native_gain=None,elapsed_s=time.monotonic()-started))


if __name__=='__main__':run()
