from dataclasses import replace
import itertools
import numpy as np
from scipy.stats import binom
from scipy.linalg import expm
from impl import synaptic_veto as v


def test_probability_bounds_and_exact_budget_for_every_allocation():
    c=v.Config(gain=1e6,k=1);q,_=v.sources();w,d,a,s=v.architecture(c)
    x=np.linspace(0,1,101)
    p=v.probability(x[None,:],c,a[:,None,:],q)
    assert p.min()>0 and p.max()<1
    np.testing.assert_allclose((p-c.p0).sum(-1),np.broadcast_to((p[0]-c.p0).sum(-1),(5,101)),atol=2e-14)
    assert set(np.flatnonzero(a[2]))=={np.argmax(s)}


def test_finite_pool_against_independent_master_equation():
    c=v.Config(pool=5,tau=.02,horizon=.05,cue=0.)
    q=.24;q1=.4;trace=v.source_trace(c,100000,87,q,q1)
    Q=np.zeros((6,6))
    for i in range(6):
        if i<5:Q[i+1,i]=(5-i)*q/c.tau
        if i>0:Q[i-1,i]=i*(1-q)/c.tau
        Q[i,i]=-Q[:,i].sum()
    expected=expm(Q*.04)@binom.pmf(np.arange(6),5,q1)
    actual=np.bincount(np.rint(trace[-1,1]*5).astype(int),minlength=6)/100000
    assert np.max(abs(actual-expected))<.004
    np.testing.assert_allclose(trace*5,np.rint(trace*5),atol=1e-14)
    assert trace.min()>=0 and trace.max()<=1


def test_shared_uniform_correlation_and_exact_majority():
    p=.45;r=.3;u=v.uniforms(np.random.default_rng(99),200000,4,r);b=u<p
    assert abs(b.mean()-p)<.003
    assert abs(np.corrcoef(b[:,0],b[:,1])[0,1]-r)<.008
    # Explicit enumeration independent of simulation for equal 4-vote majority.
    total=0.
    for bits in itertools.product((0,1),repeat=4):
        total+= (sum(bits)>=2)*p**sum(bits)*(1-p)**(4-sum(bits))
    np.testing.assert_allclose(total,binom.sf(1,4,p),atol=1e-14)
    expected=r*p+(1-r)*total
    assert abs((b.sum(1)>=2).mean()-expected)<.004


def test_zero_gain_reset_block_and_unreachable_signal():
    for c in [v.Config(gain=0),v.Config(source='reset'),v.Config(blocked=True),v.Config(chemical_delay=2.)]:
        out=v.simulate(c,n=48,seed=41)
        np.testing.assert_array_equal(out['times'][:,:,0],out['times'][:,:,1])
        assert out['budget_error']<1e-12


def test_replay_and_stochastic_classical_controls_are_exact():
    c=v.Config(tau=.2)
    a=v.simulate(c,n=48,seed=73)
    b=v.simulate(replace(c,source='replay'),n=48,seed=73,replay=a['path'])
    d=v.simulate(replace(c,source='classical_stochastic'),n=48,seed=73)
    np.testing.assert_array_equal(a['times'],b['times'])
    np.testing.assert_array_equal(a['times'],d['times'])
    np.testing.assert_array_equal(a['path'],d['path'])


def test_causality_ties_partition_and_no_stop_rebirth():
    c=v.Config(tau=.012)
    a=v.simulate(c,n=64,seed=32);t=a['times']
    assert np.all(t[...,0]>=.35-1e-12)
    assert np.all(t[...,1]>=c.cue+.04-1e-12)
    valid=np.isfinite(t[...,2]);assert np.all(t[...,2][valid]>=t[...,1][valid]+.07-1e-12)
    assert np.all(a['outcomes'].sum(-1)==1)
    tie=v.classify(np.array([[.4,.25,.4],[np.inf,np.inf,np.inf]]),1.)
    np.testing.assert_array_equal(tie,[[False,True,False],[False,False,True]])
    # Earliest possible stop activation is cue+40ms, bounding residual source age.
    exposure=a['source_exposure_after_trigger']
    max_bound=.01*np.exp(-.05/.012)/(1-np.exp(-.01/.012))
    assert exposure.max()<=max_bound+1e-12


def test_majority_ignores_influence_labels_with_equal_arrival_times():
    c=v.Config(homogeneous=True,heterogeneity=0.)
    w,d,a,score=v.architecture(c)
    assert np.all(w==w[0])
    # With equal weights, every selected k has identical one-packet vote law.
    q,_=v.sources();prob=v.probability(np.array([q+.006]),c,a,q)
    def convolution(ps):
        law=np.array([1.])
        for p in ps:law=np.convolve(law,[1-p,p])
        return law[32:].sum()
    got=[convolution(ps) for ps in prob]
    np.testing.assert_allclose(got[1:],got[1],atol=1e-14)


def test_no_detector_and_depletion_controls():
    c=v.Config(gain=1024,tau=.2,route='go')
    a=v.simulate(c,n=48,seed=82,detector_enabled=False)
    assert np.isinf(a['times'][...,1:]).all()
    assert not a['outcomes'][...,0].any()
    q,ys=v.sources();z=v.source_trace(c,10000,81,q,ys['full'],recovery='depletion')
    assert abs(z[-1,0].mean()-q*np.exp(-(.99-.2)/.2))<.0003
