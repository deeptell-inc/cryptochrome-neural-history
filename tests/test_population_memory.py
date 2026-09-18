import json,hashlib
import numpy as np
import pandas as pd
from scipy.linalg import eigh,expm
from scipy.integrate import solve_ivp
from impl import population_memory as m
from impl import dark_threshold_source as s
from impl import dark_md_coefficients as md


def case():
    d=dict(np.load(m.OUT/'latest_case.npz'))
    return d,m.population_geometry(d['H_H'],d['D_H'])


def test_pauli_rates_from_independent_noise_matrix_elements():
    data,geo=case();U=geo['U'];ops=md.operators('H');K=data['K_H']
    rotated=np.array([U.conj().T@op@U for op in ops])
    G=np.einsum('ab,aij,bij->ij',K,rotated,rotated.conj()).real
    np.fill_diagonal(G,0);np.fill_diagonal(G,-G.sum(0))
    np.testing.assert_allclose(G,geo['G'],atol=3e-12)
    assert geo['rates'].min()>0 and abs(geo['G'].sum(0)).max()<1e-12
    np.testing.assert_allclose(geo['G'],geo['G'].T,atol=1e-12)
    np.testing.assert_allclose(geo['G']@np.ones(9)/9,0,atol=1e-12)


def test_two_nucleus_factorization_and_correlation_modes():
    data,geo=case();H=data['H_H'].reshape(3,3,3,3)
    h5=np.einsum('abcb->ac',H)/3;h10=np.einsum('abad->bd',H)/3
    _,u5=eigh(h5);_,u10=eigh(h10);Uprod=np.kron(u5,u10)
    overlap=abs(geo['U'].conj().T@Uprod)**2;perm=np.argmax(overlap,axis=0)
    np.testing.assert_allclose(overlap[perm,np.arange(9)],1.,atol=1e-12)
    G=geo['G'][np.ix_(perm,perm)];T=G.reshape(3,3,3,3)
    g5=T.sum(axis=(1,3))/3;g10=T.sum(axis=(0,2))/3
    np.testing.assert_allclose(G,np.kron(g5,np.eye(3))+np.kron(np.eye(3),g10),atol=3e-12)
    rates=np.sort((-eigh(g5,eigvals_only=True))[:,None]+(-eigh(g10,eigvals_only=True))[None,:],axis=None)
    np.testing.assert_allclose(rates[1:],geo['rates'],atol=1e-12)
    chars=[m.character(s.unvec(geo['C']@geo['V'][:,i],9)) for i in range(8)]
    assert sum(x['correlation_fraction']>.999999 for x in chars)==4
    assert sum(x['N5_local_fraction']>.999999 for x in chars)==2
    assert sum(x['N10_local_fraction']>.999999 for x in chars)==2
    assert max(x['magnetization_fraction'] for x in chars)<2e-6


def test_write_read_reconstruction_and_positive_finite_difference():
    data,geo=case();z=np.load(m.OUT/'modes_and_states.npz');x=z['after_20_cycles_g0'];P=data['P_0'];r=s.trvec(9)@P
    modes=m.mode_table(geo,P,x)
    expected=float((r@(geo['P']@(x-s.X0))).real)
    assert abs(sum(a['contribution_at_zero'] for a in modes)-expected)<1e-12
    for i,row in enumerate(modes):
        v=geo['C']@geo['V'][:,i];eps=.01
        assert eigh(s.unvec(s.X0+eps*v,9),eigvals_only=True).min()>0
        delta=((r@(s.X0+eps*v))-(r@(s.X0-eps*v)))/(2*eps)
        np.testing.assert_allclose(delta.real,row['read_sensitivity'],atol=1e-13)
    assert np.linalg.norm(z['reaction_sum_g0']-z['reaction_P_weighted_g0']-z['reaction_E_weighted_g0'])<1e-13


def test_population_hold_against_independent_ode_and_full_exponential():
    data,geo=case();z=np.load(m.OUT/'modes_and_states.npz');x=z['after_20_cycles_g0']-s.X0
    p=(geo['C'].conj().T@x).real;time=.012
    ode=solve_ivp(lambda t,y:geo['G']@y,(0,10),p,rtol=1e-11,atol=1e-13,t_eval=[time,1.,10.])
    for t,y in zip(ode.t,ode.y.T):
        modal=geo['V']@(np.exp(-geo['rates']*t)*(geo['V'].T@p))
        np.testing.assert_allclose(modal,y,atol=2e-13)
    exact=expm(data['L_H']*time)@x
    spectral=m.propagate_traceless(m.wait_spectrum(data['L_H']),x,time)
    np.testing.assert_allclose(exact,spectral,atol=1e-10)
    Q=geo['Q'];C=geo['C']
    assert np.linalg.norm(Q.conj().T@data['D_H']@C)>.1
    assert np.linalg.norm(C.conj().T@data['D_H']@Q)>.1


def test_phase_average_preserves_populations_reset_erases_them():
    data,geo=case();z=np.load(m.OUT/'modes_and_states.npz');rho=s.unvec(z['before_H_wait_g0'],9);U=geo['U']
    r=U.conj().T@rho@U;average=np.zeros_like(r)
    for k in range(9):
        phase=np.exp(2j*np.pi*k*np.arange(9)/9)
        average+=phase[:,None]*r*phase.conj()[None,:]/9
    np.testing.assert_allclose(average,np.diag(np.diag(r)),atol=1e-14)
    assert np.linalg.norm(np.diag(average)-np.ones(9)/9)>1e-3
    reset=s.RESET@s.vec(rho)
    np.testing.assert_allclose(reset,s.X0,atol=1e-12)
    assert np.linalg.norm(r-np.diag(np.diag(r)))>1e-4


def test_resolvent_independent_integral_and_stiff_identity():
    data,geo=case();rate=10.;G=geo['G']
    # Integrate the Laplace-weighted Pauli propagator independently as a block ODE.
    def ode(t,x):
        A=x[:81].reshape(9,9);return np.r_[(G@A-rate*A).ravel(),(rate*A).ravel()]
    sol=solve_ivp(ode,(0,5.),np.r_[np.eye(9).ravel(),np.zeros(81)],rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(sol.y[81:,-1].reshape(9,9),m.markov_wait(G,rate),atol=1e-11)
    WH=m.unital_wait(data['L_H'],10.)
    np.testing.assert_allclose(WH@s.X0,s.X0,atol=1e-14)
    np.testing.assert_allclose(s.trvec(9)@WH,s.trvec(9),atol=1e-13)
    assert json.loads((m.OUT/'independent_resolvent.json').read_text())['baseline_yield_disagreement']<1e-8


def test_no_bath_counterexample_independent_unitary_wait():
    data,geo=case();saved=np.load(m.OUT/'no_bath_counterexample.npz')
    def analytic_wait(H,rate):
        ev,U=eigh(H);basis=np.kron(U.conj(),U)
        factor=s.vec(rate/(rate+1j*(ev[:,None]-ev[None,:])))
        return (basis*factor)@basis.conj().T
    for state,rate in [('R',10.),('H',10.),('E',4.)]:
        L=s.commutator_generator(data['H_'+state])
        np.testing.assert_allclose(m.unital_wait(L,rate),analytic_wait(data['H_'+state],rate),atol=1e-8)
    d=pd.read_csv(m.OUT/'no_bath_reaction_memory.csv')
    assert .95<d.query('additional_cycles==1').relative_memory.iloc[0]<.99
    assert d.query('additional_cycles==10').relative_memory.iloc[0]<.8
    np.testing.assert_allclose(s.trvec(9)@saved['cycle'],s.trvec(9),atol=1e-10)


def test_channel_physicality_shared_models_and_frozen_inputs():
    data,geo=case()
    for kwargs in ({},{'retention':0.},{'retention':0.,'reset_at':'reaction'},{'noise_scale':100.}):
        M,_=m.cycle(data,**kwargs)
        np.testing.assert_allclose(s.trvec(9)@M,s.trvec(9),atol=1e-10)
        choi=M.reshape(9,9,9,9,order='F').transpose(0,2,1,3).reshape(81,81,order='F')
        assert eigh((choi+choi.conj().T)/2,eigvals_only=True).min()>-1e-8
        rows,_=m.compare_cycle(M,data['P_0'],geo)
        assert all(x['minimum_eigenvalue']>0 for x in rows)
    scan=pd.read_csv(m.OUT/'factor_scans.csv');assert abs(scan.query('model=="reset"').delta_yield).max()<1e-10
    old=json.loads((m.ROOT/'investigations/2026-09-16-population-memory/preserved_inputs.json').read_text())['records']
    assert len(old)==1339
    for x in old:assert hashlib.sha256((m.ROOT/x['path']).read_bytes()).hexdigest()==x['sha256'],x['path']
    scope=json.loads((m.OUT/'implementation_scope.json').read_text())
    assert scope['native_population_generator'] is None and scope['native_reset_probability'] is None


def test_escape_no_nine_population_closure():
    data,geo=case();extract=geo['C'].conj().T@s.TRACE_E
    op=np.kron(s.S[0],s.NI[0][0]@s.NI[0][0]-s.NI[0][1]@s.NI[0][1]);op/=np.linalg.norm(op)
    delta=.02*s.vec(op)
    assert np.linalg.norm(extract@delta)<1e-14
    assert np.linalg.norm(extract@data['L_E']@delta)>1e4
    assert eigh(np.eye(18)/18+.01*op,eigvals_only=True).min()>0
    saved=json.loads((m.OUT/'escape_closure.json').read_text())
    np.testing.assert_allclose(np.linalg.norm(extract@data['L_E']@delta),saved['different_population_derivative_s'],rtol=1e-12)
