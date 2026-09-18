import numpy as np
import pytest
from scipy.integrate import solve_ivp
from impl import chemical_carrier_mapping as c

def test_units_and_correlated_effective_count():
    assert np.isclose(c.molecule_count(1,.01),6.02214076)
    assert c.effective_sites(400,0)==400
    assert np.isclose(c.effective_sites(400,.01),80.1603206413)
    assert c.effective_sites(400,1)==1

def test_peroxide_reconvergence_against_direct_ODE():
    for q in [.238386607448808,.244088539901755]:
        sol=solve_ivp(lambda t,y:[-y[0],y[0]-5*y[1],y[0]],(0,2),[(1-q)/2,(1+q)/2,(1+q)/2],rtol=1e-11,atol=1e-13)
        actual=c.peroxide_path(q,2,1,5,direct_delivery=1,early_escape_delivery=1,late_escape_delivery=1)
        np.testing.assert_allclose(actual,sol.y[:,-1],atol=1e-11)
        end=c.peroxide_path(q,100,1,5,direct_delivery=1,early_escape_delivery=1,late_escape_delivery=1)
        assert np.isclose(end[2],1)
    # Equal eventual yield, but early versus late available-concentration contrast reverses.
    def delta(t):return c.peroxide_path(.25,t,1,5,direct_delivery=1,early_escape_delivery=1,late_escape_delivery=1)[1]-c.peroxide_path(.24,t,1,5,direct_delivery=1,early_escape_delivery=1,late_escape_delivery=1)[1]
    assert delta(0)>0 and delta(2)<0

def test_three_state_generator_recovery_and_independent_solution():
    a,b,h,o=3.,.2,10.,1.
    G=c.cofactor_generator(a,b,h,o)
    np.testing.assert_allclose(G.sum(0),0,atol=1e-14)
    x=c.cofactor_step([1,0,0],.2,write_s=a,off_s=b,bind_reduced_s=h,bind_oxidized_s=o)
    sol=solve_ivp(lambda t,y:G@y,(0,.2),[1.,0.,0.],rtol=1e-11,atol=1e-13)
    np.testing.assert_allclose(x,sol.y[:,-1],atol=1e-11)
    assert np.min(x)>=0 and np.isclose(x.sum(),1)
    # Independent backward first-passage equations for transient O,U states.
    T=np.linalg.solve(np.array([[b,-b],[-o,h+o]]),np.ones(2))[0]
    assert np.isclose(T,c.recovery_mfpt(b,h,o))

def test_fast_rebinding_recovers_two_state_limit_and_chemical_gain():
    z=c.cofactor_step([1,0,0],.2,write_s=2,off_s=1,bind_reduced_s=1e7,bind_oxidized_s=0)[1]
    assert np.isclose(z,2/3*(1-np.exp(-.6)),atol=1e-7)
    assert c.write_rate(10,kcat_s=4,KM_uM=10)==2

def test_probability_gain_dimensions_and_unknown_rejection():
    beta=128/(.45*.55);eps=1e-7
    slope=(c.release_probability(.3+eps,.3,p0=.45,beta=beta)-c.release_probability(.3-eps,.3,p0=.45,beta=beta))/(2*eps)
    assert np.isclose(slope,128)
    d=c.required_occupancy_delta(.01,p0=.45,beta=beta)
    assert np.isclose(c.release_probability(.3+d,.3,p0=.45,beta=beta),.46)
    with pytest.raises(ValueError):c.require_identified({'kcat_s':None},['kcat_s'])
    assert c.cofactor_release([0,0,1],[1,0,0],p0=.45,beta_oxidized=10,beta_apo=0)==pytest.approx(.45)
    assert c.cofactor_release([0,0,1],[1,0,0],p0=.45,beta_oxidized=10,beta_apo=1)>.45

def test_finite_bolus_mass_balance_and_blockade():
    kw=dict(sites=20,carbonyls=6,volume_um3=.01,k2_uM_s=.1,off_s=1.,bind_reduced_s=100.,clearance_s=.5,n=128,seed=419,times_s=[.2,2.2])
    x=c.finite_carbonyl_trials(**kw)
    assert np.all(x[:,:,2]<=6)
    assert np.all(x[:,:,4]<=6)
    z=c.finite_carbonyl_trials(**dict(kw,k2_uM_s=0))
    assert np.all(z[:,:,2:5]==0)
