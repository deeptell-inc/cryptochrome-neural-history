import json
import numpy as np
from scipy.integrate import solve_ivp
from impl import sparse_synaptic as s

def test_allocation_equal_budget_and_selective_block():
    w,t=s.network();score=s.influence(w,t,'go');a,ids=s.allocations(score)
    np.testing.assert_allclose(a[1:6].sum(1),1,atol=1e-15)
    assert all(np.count_nonzero(a[k])==8 for k in (2,3,4,5))
    assert np.array_equal(a[8],a[1])
    for k in (6,7,9):assert np.isclose(a[k].sum(),.875)
    assert np.all(a[6,ids['top']]==0) and np.all(a[9,ids['absolute']]==0)

def test_exact_transition_against_independent_ode():
    A,d,sigma,*_=s.system('go');F,c,Q=s.transition(A,d,sigma,.03)
    def fun(t,y):
        mean=y[:2];cov=y[2:].reshape(2,2)
        return np.r_[A@mean+d,(A@cov+cov@A.T+np.diag(sigma**2)).ravel()]
    initial=np.r_[.2,-.1,np.zeros(4)]
    z=solve_ivp(fun,(0,.03),initial,rtol=1e-11,atol=1e-13).y[:,-1]
    np.testing.assert_allclose(z[:2],F@initial[:2]+c,atol=1e-11)
    np.testing.assert_allclose(z[2:].reshape(2,2),Q,atol=1e-11)

def test_probability_null_causality_and_rescue():
    dy=s.source_values()['full']['delta_yield']
    z=s.simulate('go',dy,512,n=128)
    assert np.array_equal(z['rt'][1],z['rt'][8])
    assert np.array_equal(z['aggregate'][:,1],z['aggregate'][:,8])
    assert np.max(abs(z['aggregate'][:100]-z['aggregate'][:100,0,None]))==0
    null=s.simulate('go',0.,512,n=128)
    assert np.all(null['rt']==null['rt'][0])
    assert all(x['delta_RMST_s']==0 for x in null['records'])

def test_coupled_grid_endpoint_and_crossing_error():
    dy=s.source_values()['full']['delta_yield']
    for route in ('go','trigger','stop'):
        fine=s.simulate(route,dy,512,n=128,dt=.001)
        coarse=s.simulate(route,dy,512,n=128,dt=.002)
        np.testing.assert_allclose(coarse['aggregate'],fine['aggregate'][1::2],atol=1e-12)
        np.testing.assert_allclose(coarse['analytic'],fine['analytic'][1::2],atol=1e-12)
        # Coarse sampling can miss a transient fine crossing; never claim an upper2ms bound.
        assert np.all(coarse['rt']>=fine['rt'])

def test_route_block_and_homogeneous_counterexample():
    dy=s.source_values()['full']['delta_yield']
    z=s.simulate('go',dy,512,n=128,coupling=0.)
    assert np.array_equal(z['rt'][3],z['rt'][0])
    assert z['records'][3]['analytic_delta_state_cue_aligned']>0
    h=s.simulate('go',dy,512,n=128,homogeneous=True)
    np.testing.assert_allclose(h['analytic'][:,1:6],np.repeat(h['analytic'][:,1,None],5,axis=1),atol=1e-12)

def test_completed_data_against_analytic_mean_and_race_partition():
    import pandas as pd
    d=pd.read_csv(s.OUT/'neural_summary.csv');r=pd.read_csv(s.OUT/'race_summary.csv')
    p=d.query('run=="primary" and route=="go"')
    error=abs(p.delta_state_cue_aligned-p.analytic_delta_state_cue_aligned)
    assert np.all(error<=5*p.MC_SE_cue_state+1e-7)
    assert d.probability_min.min()>=0 and d.probability_max.max()<=1
    np.testing.assert_allclose(r.response_probability+r.cancel_probability+r.unresolved_probability,1,atol=1e-12)
    null=d.query('run=="reset"');assert abs(null.delta_RMST_s).max()==0
    assert abs(null.delta_state_cue_aligned).max()==0
    for _,g in p.groupby(['seed','gain_probability_per_yield']):
        a=g.set_index('arm');assert a.loc['targeted','analytic_delta_state_cue_aligned']>a.loc['diffuse','analytic_delta_state_cue_aligned']
        np.testing.assert_allclose(a.loc[['diffuse','random_sparse','targeted','weak','opposite'],'expected_extra_release_count'],a.loc['diffuse','expected_extra_release_count'],atol=1e-12)

def test_classical_replay_observational_equivalence():
    # The classical circuit's transition law depends on release probabilities,
    # not on the microscopic label of the source producing them.
    dy=s.source_values()['full']['delta_yield']
    microscopic=s.simulate('trigger',dy,512,n=128,seed=82)
    decay=solve_ivp(lambda t,m:-m/.012,(0,.30),[dy],rtol=1e-11,atol=1e-14,dense_output=True)
    classical=s.simulate('trigger',dy,512,n=128,seed=82,profile=lambda t:decay.sol(t)[0])
    assert np.array_equal(microscopic['rt'],classical['rt'])
    assert np.array_equal(microscopic['aggregate'],classical['aggregate'])
