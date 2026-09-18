"""Conditional nuclear population memory with frozen upstream chemistry."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from scipy.linalg import eigh,expm,null_space
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl import population_memory as m
from impl import dark_threshold_source as s
from impl import dark_md_coefficients as md
from impl import dark_finite_spike as neuron
HERE=Path(__file__).resolve().parent

def save(name,x):s.save(m.OUT/name,x)
def csv(name,x):pd.DataFrame(x).to_csv(m.OUT/name,index=False)

def main():
    m.OUT.mkdir(exist_ok=True)
    old=json.loads((HERE/'preserved_inputs.json').read_text())['records']
    for x in old:assert hashlib.sha256((ROOT/x['path']).read_bytes()).hexdigest()==x['sha256'],x['path']
    cache=m.OUT/'latest_case.npz'
    if not cache.exists():np.savez_compressed(cache,**m.build_case())
    data=dict(np.load(cache));geo=m.population_geometry(data['H_H'],data['D_H'])
    C,V,G=geo['C'],geo['V'],geo['G'];DP=geo['P'];r=s.trvec(9)@data['P_0']
    baselineM,parts=m.cycle(data);baseline,states=m.compare_cycle(baselineM,data['P_0'],geo)
    csv('baseline_comparison.csv',baseline)
    population_cycle=(C.conj().T@baselineM@C).real
    csv('cycle_population_channel.csv',[dict(destination=i,source=j,probability=population_cycle[i,j]) for i in range(9) for j in range(9)])
    eig_cycle=np.linalg.eigvals(population_cycle)
    csv('cycle_eigenvalues.csv',[dict(real=float(x.real),imag=float(x.imag),modulus=float(abs(x)),
          log_attenuation_per_cycle=float(-np.log(abs(x))) if abs(x)>0 else None,
          unit='per completed cycle, NOT s^-1; no single cycle duration assigned') for x in eig_cycle])
    frozen=pd.read_csv(ROOT/'data/dark-basis-resolution/conditional_products_spikes.csv').query('condition=="upcj2N__full" and gamma_oxygen_s==0')
    check=[]
    for row in baseline:
        name='scalar' if row['model']=='reset' else row['model']
        oldrow=frozen.query('model==@name').iloc[0]
        check.append(dict(model=name,yield_difference=row['yield_P']-oldrow.yield_P))
    csv('frozen_reproduction.csv',check)
    assert max(abs(x['yield_difference']) for x in check)<1e-10
    writing=[];snapshots={};basisrows=[]
    for gamma in (0.,1e8):
        P,E=data[f'P_{gamma:g}'],data[f'Escape_{gamma:g}'];cyc,pieces=m.cycle(data,gamma=gamma)
        stages={'reaction_P_weighted':P@s.X0,'reaction_E_weighted':s.TRACE_E@E@s.X0,
                'reaction_sum':(P+s.TRACE_E@E)@s.X0,'before_H_wait':pieces['before_H']@s.X0,
                'after_1_cycle':cyc@s.X0,'after_20_cycles':np.linalg.matrix_power(cyc,20)@s.X0}
        for stage,x in stages.items():
            mass=float((s.trvec(9)@x).real);shift=x-mass*s.X0
            # For branch states mode_table's constant subtraction has zero traceless projection.
            rows=m.mode_table(geo,P,x)
            for row in rows:writing.append(dict(gamma_oxygen_s=gamma,stage=stage,branch_mass=mass,**row))
            snapshots[f'{stage}_g{gamma:g}']=x
        for j in range(9):
            basisrows.append(dict(gamma_oxygen_s=gamma,HQ_level=j,energy_rad_s=geo['energies'][j],
                                  readout_yield=float((s.trvec(9)@P@C[:,j]).real),
                                  initial_population=1/9,
                                  after_1=float((C[:,j].conj()@stages['after_1_cycle']).real),
                                  after_20=float((C[:,j].conj()@stages['after_20_cycles']).real)))
    csv('write_read_modes.csv',writing);csv('HQ_levels.csv',basisrows)
    np.savez_compressed(m.OUT/'modes_and_states.npz',G=G,U=geo['U'],C=C,V=V,rates=geo['rates'],cycle=baselineM,**snapshots)
    # Nuclear marginals versus their connected correlation, distinct from linear sectors.
    rho=s.unvec(states['population_HQ'],9);x=rho.reshape(3,3,3,3)
    rho5=np.einsum('abcb->ac',x);rho10=np.einsum('abad->bd',x)
    product=np.kron(rho5,rho10);connected=rho-product
    save('correlations.json',dict(connected_HS_norm=float(np.linalg.norm(connected)),
         connected_yield=float((r@s.vec(connected)).real),
         product_of_marginals_yield=float((r@s.vec(product)).real),
         full_population_yield=float((r@states['population_HQ']).real),
         product_state_min=float(eigh(product,eigvals_only=True).min()),
         interpretation='HQ Hamiltonian has no internuclear interaction; diagonal joint states are separable classical correlations, not entanglement'))
    # Spin-insensitive preparation instrument, same reference branch mass and chemical waits.
    blind,_=m.cycle(data,spinblind=True);bx=np.linalg.matrix_power(blind,20)@s.X0
    save('spinblind_write_control.json',dict(delta_yield_same_quantum_probe=float((r@(bx-s.X0)).real),
         state_distance_uniform=float(np.linalg.norm(bx-s.X0)),
         scope='preparation-only control removes nuclear selection and electron-nuclear correlations; reference branch fractions preserved; not all-quantum-off chemistry'))
    generator_rows=[];coupling=[]
    for state in ('R','H'):
        geom=m.population_geometry(data['H_'+state],data['D_'+state]);CC,QQ=geom['C'],geom['Q'];L=data['L_'+state];D=data['D_'+state]
        for j,rate in enumerate(geom['rates']):
            generator_rows.append(dict(state=state,mode=j+1,decay_rate_s=rate,time_constant_s=1/rate,
                 **m.character(s.unvec(CC@geom['V'][:,j],9))))
        coupling.append(dict(state=state,population_from_coherence_Fnorm_s=float(np.linalg.norm(CC.conj().T@D@QQ)),
             coherence_from_population_Fnorm_s=float(np.linalg.norm(QQ.conj().T@D@CC)),
             D_Fnorm_s=float(np.linalg.norm(D)),minimum_energy_gap_rad_s=float(np.min(np.diff(geom['energies']))),
             G_column_sum_error=float(abs(geom['G'].sum(0)).max()),G_symmetry_error=float(abs(geom['G']-geom['G'].T).max()),
             G_roundoff_adjustment=geom['roundoff_adjustment']))
    csv('state_relaxation_modes.csv',generator_rows);csv('generator_coupling.csv',coupling)
    Q=geo['Q'];raw_reaction=data['P_0']+s.TRACE_E@data['Escape_0']
    mixing=[]
    for label,channel in [('reaction_nuclear_marginal',raw_reaction),('completed_cycle',baselineM)]:
        mixing.append(dict(stage=label,population_to_coherence=float(np.linalg.norm(Q.conj().T@channel@C)),
                           coherence_to_population=float(np.linalg.norm(C.conj().T@channel@Q))))
    csv('reaction_cycle_coupling.csv',mixing)
    extract=C.conj().T@s.TRACE_E
    inject=np.column_stack([s.vec(np.kron(np.eye(2)/2,s.unvec(x,9))) for x in C.T])
    reduced=extract@data['L_E']@inject
    defect=extract@data['L_E']-reduced@extract
    op=np.kron(s.S[0],s.NI[0][0]@s.NI[0][0]-s.NI[0][1]@s.NI[0][1]);op/=np.linalg.norm(op)
    rho_plus=np.eye(18)/18+.01*op;rho_minus=np.eye(18)/18-.01*op
    save('escape_closure.json',dict(
        nine_population_closure_defect_Fnorm_s=float(np.linalg.norm(defect)),
        same_initial_nuclear_population_difference=float(np.linalg.norm(extract@s.vec(rho_plus-rho_minus))),
        different_population_derivative_s=float(np.linalg.norm(extract@data['L_E']@s.vec(rho_plus-rho_minus))),
        example_minimum_eigenvalue=float(min(eigh(rho_plus,eigvals_only=True).min(),eigh(rho_minus,eigvals_only=True).min())),
        probe_coherence_sensitivity_norm=float(np.linalg.norm(r@Q)),
        scope='two valid E density matrices with identical nuclear populations but different electron-nuclear correlations; mathematical closure counterexample, not measured states'))
    # Direct resolvent comparison: projecting exact wait vs first projecting generator.
    waits=[]
    for state in ('R','H'):
        ge=m.population_geometry(data['H_'+state],data['D_'+state]);CC=ge['C']
        for rate in (.1,10.,1e3,1e6,1e8):
            W=md.wait_map(data['L_'+state],rate)
            projected=(CC.conj().T@W@CC).real;pauli=m.markov_wait(ge['G'],rate)
            waits.append(dict(state=state,rate_s=rate,max_population_wait_error=float(abs(projected-pauli).max())))
    csv('projected_wait_audit.csv',waits)
    # Independent numerical route, fixing the exact invariant identity of waits.
    WR=m.unital_wait(data['L_R'],10.);WH=m.unital_wait(data['L_H'],10.);WE=m.unital_wait(data['L_E'],4.)
    Mstable=WH@WR@(WR@data['P_0']+s.TRACE_E@WE@data['Escape_0'])
    stable_rows,stable_states=m.compare_cycle(Mstable,data['P_0'],geo)
    stable_delta=float((r@(stable_states['full']-s.X0)).real)
    save('independent_resolvent.json',dict(full_delta_yield=stable_delta,
         baseline_yield_disagreement=float(abs(stable_delta-baseline[0]['delta_yield'])),
         max_cycle_entry_disagreement=float(abs(Mstable-baselineM).max()),
         uniform_H_wait_error=float(np.linalg.norm(WH@s.X0-s.X0)),
         uniform_E_wait_error=float(np.linalg.norm(WE@s.vec(np.eye(18)/18)-s.vec(np.eye(18)/18))),
         interpretation='exact unital identity separated before linear solve; discrepancy is a numerical diagnostic, not physical memory'))
    print('write, mode and generator audits complete',flush=True)
    # Isolated HQ pulse-chase: no recovery or further reactions during the delay.
    delta=states['full']-s.X0
    pdelta=DP@delta;w=V.T@(C.conj().T@delta).real;read=(r@C@V).real
    spectrum=m.wait_spectrum(data['L_H']);spectrum_noH=m.wait_spectrum(data['D_H'])
    holds=[]
    for time in np.r_[0.,np.logspace(-8,2,81)]:
        exact=m.propagate_traceless(spectrum,delta,time)
        avg=m.propagate_traceless(spectrum,pdelta,time)
        pop=C@(V@(np.exp(-geo['rates']*time)*w))
        holds.append(dict(time_s=time,full_yield_memory=float((r@exact).real),
            phase_averaged_start_yield_memory=float((r@avg).real),population_yield_memory=float((r@pop).real),
            mode_sum_yield=float(np.sum(w*read*np.exp(-geo['rates']*time))),
            population_L2=float(np.linalg.norm(C.conj().T@exact)),coherence_L2=float(np.linalg.norm((np.eye(81)-DP)@exact)),
            reset_yield_memory=0.))
    csv('HQ_pulse_chase.csv',holds)
    base=float(w@read)
    crossing={}
    for fraction in (.5,1/np.e,.1):
        crossing[str(fraction)]=brentq(lambda t:np.sum(w*read*np.exp(-geo['rates']*t))-fraction*base,0,300.)
    save('readable_lifetime.json',dict(definition='isolated HQ hold, positive weighted multi-exponential readout; no re-reaction or chemical state change',
        threshold_times_s=crossing,readout_weighted_initial_rate_s=float(np.sum(w*read*geo['rates'])/base),
        modes_all_contributions_positive=bool(np.all(w*read>0)),native_lifetime_s=None))
    # Strict Pauli-only change isolates population relaxation from dephasing.
    scaled=[]
    for scale in (0.,.01,.1,1.,10.,100.,1000.):
        for time in (.012,.1,1.,10.):
            value=float(np.sum(w*read*np.exp(-scale*geo['rates']*time)))
            scaled.append(dict(population_rate_scale=scale,time_s=time,readable_fraction=value/base))
    csv('isolated_population_rate_scan.csv',scaled)
    # Chemical changes and closed-shell bath changes in the SAME within-cycle full model.
    configs=[('baseline',1.,{})]
    for a in ('a','b','cp','ce'):
        for rate in (.1,1.,10.,100.,1e4,1e6,1e8):configs.append((a,rate,{a:rate}))
    for q in (0.,.1,.5,.9,1.):
        for at in ('reaction','reduction'):configs.append(('retention_'+at,q,dict(retention=q,reset_at=at)))
    for scale in (0.,.01,.1,1.,10.,100.,1000.):configs.append(('closed_shell_noise_scale',scale,dict(noise_scale=scale)))
    for axis in ('h_noise_scale','r_noise_scale','escape_noise_scale'):
        for scale in (0.,.01,.1,1.,10.,100.):configs.append((axis,scale,{axis:scale}))
    for kappa in (0.,1.,1e3,1e6,1e9):configs.append(('HQ_phase_rate',kappa,dict(phase_rate=kappa)))
    for rate in (1e4,1e6,1e8):configs.append(('all_waits',rate,dict(a=rate,b=rate,cp=rate,ce=rate)))
    scans=[];renewal=[]
    for axis,value,kw in configs:
        cyc,pieces=m.cycle(data,**kw);rows,xx=m.compare_cycle(cyc,data['P_0'],geo)
        yp=float((r@s.X0).real)
        duration=1/kw.get('a',10)+1/kw.get('b',10)+yp/kw.get('cp',10)+(1-yp)/kw.get('ce',4)
        for row in rows:scans.append(dict(axis=axis,value=value,mean_first_cycle_duration_s=duration,
                           instantaneous_reaction_approx_valid=bool(max(kw.get(z,v) for z,v in [('a',10),('b',10),('cp',10),('ce',4)])<4e4),**row))
        # Propagate the SAME pre-existing contrast for both histories; no new difference written.
        if axis in ('baseline','retention_reduction','retention_reaction','closed_shell_noise_scale','h_noise_scale','r_noise_scale','escape_noise_scale'):
            d=delta.copy()
            for n in (0,1,2,5,10):
                d=np.linalg.matrix_power(cyc,n)@delta
                renewal.append(dict(axis=axis,value=value,additional_cycles=n,
                                    delta_probe_yield=float((r@d).real),relative_memory=float((r@d).real/(r@delta).real)))
    csv('factor_scans.csv',scans);csv('reaction_renewal_memory.csv',renewal)
    nb=m.no_bath_case(data)
    # Use invariant-preserving resolvents for zero-dissipator, very stiff waits.
    wr=m.unital_wait(nb['L_R'],10.);wh=m.unital_wait(nb['L_H'],10.);we=m.unital_wait(nb['L_E'],4.)
    cyc_nb=wh@wr@(wr@nb['P_0']+s.TRACE_E@we@nb['Escape_0'])
    rnb=s.trvec(9)@nb['P_0'];memory_nb=[]
    for n in (0,1,2,5,10):
        propagated=np.linalg.matrix_power(cyc_nb,n)@delta
        memory_nb.append(dict(additional_cycles=n,delta_probe_yield=float((rnb@propagated).real),
                             relative_memory=float((rnb@propagated).real/(rnb@delta).real)))
    csv('no_bath_reaction_memory.csv',memory_nb)
    np.savez_compressed(m.OUT/'no_bath_counterexample.npz',cycle=cyc_nb,product=nb['P_0'])
    print('hold, reset and chemical scans complete',flush=True)
    # Phase randomization and population loss are distinct controls.
    pre=parts['before_H']@s.X0;preD=DP@pre
    E,U=geo['energies'],geo['U'];rho=U.conj().T@s.unvec(pre,9)@U
    phases=[]
    for t in np.linspace(0,10e-6,401):
        unitary=U@(np.exp(-1j*(E[:,None]-E[None,:])*t)*rho)@U.conj().T
        full=float((r@s.vec(unitary)).real);pop=float((r@preD).real)
        phases.append(dict(time_s=t,coherent_yield=full,phase_averaged_yield=pop,
                           coherence_yield_part=full-pop,population_difference_norm=float(np.linalg.norm((C.conj().T@s.vec(unitary))-(C.conj().T@preD)))))
    csv('phase_only_control.csv',phases)
    # No-H diagnostic: coupling need not be negligible if spectral separation disappears.
    nogap=[]
    for t in (.012,.1,1.,10.):
        exact=m.propagate_traceless(spectrum_noH,pdelta,t)
        pauli=C@(V@(np.exp(-geo['rates']*t)*w))
        nogap.append(dict(time_s=t,exact_D_only_yield=float((r@exact).real),pauli_yield=float((r@pauli).real),
                          difference=float((r@(exact-pauli)).real),
                          generated_coherence_norm=float(np.linalg.norm((np.eye(81)-DP)@exact)),
                          scope='zero Hamiltonian while keeping dissipator and chosen HQ axes fixed: mathematical nonsecular counterexample, not CRY parameter estimate'))
    csv('nonsecular_counterexample.csv',nogap)
    # Shared downstream transfer for representative interventions only.
    selected=[('baseline',{}),('population_noise_x100',dict(noise_scale=100.)),('reduction_reset',dict(retention=0.)),
              ('reaction_reset',dict(retention=0.,reset_at='reaction')),('HQ_wait_slow',dict(b=.1)),('HQ_phase_fast',dict(phase_rate=1e9))]
    neural=[]
    for label,kw in selected:
        cyc,_=m.cycle(data,**kw);rows,_=m.compare_cycle(cyc,data['P_0'],geo)
        yref=float((r@s.X0).real);ref=neuron.first_passage(.05*yref,.05*yref,.012,'tonic_excess',n=1200)['firing_probability']
        for row in rows:
            y=row['yield_P'];prob=neuron.first_passage(.05*y,.05*yref,.012,'tonic_excess',n=1200)['firing_probability']
            neural.append(dict(condition=label,model=row['model'],delta_yield=row['delta_yield'],delta_spike_probability=prob-ref,
                               source='shared frozen gain100pA, burst.05, excess12ms, no leak 50ms model; uncalibrated'))
    csv('shared_neural_comparison.csv',neural)
    # Rank/class audit: E has electron-nuclear correlations, no autonomous 9-state nuclear generator.
    save('implementation_scope.json',dict(
        boundary_population='DP_HQ M_full DP_HQ; quantum RP and full recovery dynamics shared',
        pauli='G=C^dag D C, column convention, symmetric unital high-temperature white-noise model',
        nonsecular='D_PC and D_CP nonzero; phases suppress their effect at current nondegenerate energy gaps',
        E='18-state electron-nuclear Hilbert space; partial nuclear trace generally not a 9-state Markov semigroup',
        old_global_gamma='dark_threshold_source.exponential_wait: gamma(RESET-I) assigns the SAME decay gamma to all eight traceless populations and 72 coherences; not a resolved measured T1',
        old_memory='dark_md_coefficients.memory returns Frobenius RMS survival of initial subspaces, including conversion between them, not readout-weighted T1/T2',
        basis='mean HQ Hamiltonian, two spin-1 14N (N5/N10), quadrupolar plus Zeeman; no internuclear Hamiltonian coupling',
        native_population_generator=None,native_chemical_rates=None,native_reset_probability=None,native_wait_distribution=None,native_neural_gain=None,
        source='latest upcj2N__full SQ; R/H previous embedded tensors; oxidized dense6ps trajectory used counterfactually in reduced states; 1ps white spectral window',
        warnings=['all rates conditional, not physiological confidence intervals','cycles not fixed chronological preparation','fast chemical waits approaching RP rate violate instantaneous-reaction separation and are mathematical controls','MD does not calibrate seconds-long native relaxation']))
    print('population memory audit complete',flush=True)

if __name__=='__main__':main()
