from dataclasses import replace
import numpy as np
from impl import stochastic_branch_spike as s
from impl import branch_current_gain as b

Y0=.238386607448808;DY=.0057019324529471


def test_count_transition_marginals_and_branch_conservation():
    rng=np.random.default_rng(81);n=60000;N=50
    a,c=s.pair_update(rng,np.full(n,10),np.full(n,16),N,.3,.3,.7)
    for arr,old in [(a,10),(c,16)]:
        pp=.7+.3*.3;pe=.3*.3;mean=old*pp+(N-old)*pe
        var=old*pp*(1-pp)+(N-old)*pe*(1-pe)
        assert abs(arr.mean()-mean)<5*np.sqrt(var/n)
        assert abs(arr.var()-var)<.15
        assert np.all((arr>=0)&(arr<=N))
        assert np.all(arr+(N-arr)==N)
    assert np.all(c>=a)


def test_stationary_covariance_matches_two_state_generator():
    rng=np.random.default_rng(123)
    traces=s.count_traces(200,.5,.002,.0002,1500,48,rng)
    fit=s.fit_count_noise(traces,.5,.0002,5)
    assert abs(fit['effective_count']/200-1)<.06
    assert abs(fit['tau_s']/.002-1)<.08


def test_count_kernel_matches_independent_master_equation():
    from scipy.linalg import expm
    from scipy.stats import binom
    N=4;q=.3;tau=.012;dt=.004
    L=np.zeros((N+1,N+1))
    for k in range(N+1):
        if k<N:L[k+1,k]=(N-k)*q/tau
        if k>0:L[k-1,k]=k*(1-q)/tau
        L[k,k]=-L[:,k].sum()
    transition=expm(L*dt);decay=np.exp(-dt/tau)
    for k in range(N+1):
        on=binom.pmf(np.arange(k+1),k,decay+q*(1-decay))
        off=binom.pmf(np.arange(N-k+1),N-k,q*(1-decay))
        assert np.max(abs(transition[:,k]-np.convolve(on,off)))<1e-13
    stationary=binom.pmf(np.arange(N+1),N,q)
    flux=L*stationary[None,:]
    assert np.max(abs(flux-flux.T))<1e-13


def test_shared_nulls_and_censoring():
    p=b.Scenario(tau_gate_s=.005,eta_E=1)
    tr=s.simulate_pair(Y0,DY,p,n=256,seed=33,noise=replace(s.Noise(),burn_s=.01))
    assert np.array_equal(tr['first_s'][0],tr['first_s'][1])
    tr=s.simulate_pair(Y0,0,replace(p,eta_E=0),n=256,seed=33,noise=replace(s.Noise(),burn_s=.01))
    assert np.array_equal(tr['first_s'][0],tr['first_s'][1])
    d=s.distribution(np.array([.02,np.inf]),.05)
    assert d['probability']==.5 and d['RMST_ms']==35


def test_deterministic_leaky_step_and_refinement():
    p=b.Scenario(capacity_nS=0,tau_gate_s=.005)
    noise=replace(s.Noise(),current_pA=12,background_sigma_mV_sqrt_s=0,burn_s=0)
    tr=s.simulate_pair(Y0,DY,p,n=2,seed=11,noise=noise,molecular=False,channel=False,dt=.00005)
    exact=-.02*np.log(1-10/12)
    assert np.max(abs(tr['first_s']-exact))<.00005
    noise=replace(noise,current_pA=0)
    tr=s.simulate_pair(Y0,DY,replace(p,capacity_nS=.2),n=2,seed=11,noise=noise,molecular=False,channel=False,dt=.00005)
    # Voltage statistics agree across identical deterministic copies; no spikes.
    assert np.all(~np.isfinite(tr['first_s']))
    assert np.max(tr['moments'][:,2:4])<1e-12
