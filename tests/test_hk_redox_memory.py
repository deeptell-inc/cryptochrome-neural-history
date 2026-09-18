import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from impl.hk_redox_memory import (EffectiveMemory, two_state_step, a_current,
    current_change_bounds, voltage_exchange, require_branch_gain)


def test_exact_chemical_step_against_generator_and_limits():
    for a,b,dt in [(0,0,5),(.1,.3,20),(1,0,100),(0,1,10)]:
        Q=np.array([[-a,b],[a,-b]])
        for z in [0.,.2,1.]:
            prob=expm(Q*dt)@np.array([1-z,z])
            result=two_state_step(z,a,b,dt)
            assert prob.sum()==pytest.approx(1.)
            assert result==pytest.approx(prob[1],abs=1e-13)
            assert 0<=result<=1
    with pytest.raises(ValueError):two_state_step(-.1,1,1,1)
    with pytest.raises(ValueError):two_state_step(.1,np.nan,1,1)


def test_scaled_occupation_families_and_independent_ode():
    p=EffectiveMemory(.03,.1,.6,.3,.8)
    for D in [.01,.2,.6]:
        rates=p.physical_realization(D);z=rates['z0'];outputs=[z]
        for start,end,extra in [(0,10,0),(10,30,.1),(30,40,0)]:
            sol=solve_ivp(lambda t,x:rates['oxidation_per_min']*(1-x)-(rates['exchange_rest_per_min']+extra)*x,
                          (start,end),[z],rtol=1e-11,atol=1e-13)
            z=sol.y[0,-1];outputs.append(z)
        scaled=(np.array(outputs)-rates['z0'])/D
        np.testing.assert_allclose(scaled,p.coordinate([0,10,30,40],True),atol=3e-10)
        assert rates['added_oxidation_per_min']>=0
    with pytest.raises(ValueError):p.physical_realization(.7)


def test_zero_input_train_block_and_causality():
    p=EffectiveMemory(.05,0.,1.,.3,.7)
    np.testing.assert_array_equal(p.coordinate([0,5,10],True),p.coordinate([0,5,10],False))
    np.testing.assert_allclose(p.coordinate([10,20,40],True),p.coordinate([10,20,40],False))
    q=EffectiveMemory(.05,.2,1.,0.,0.)
    assert np.all(q.log_tau_ratio(np.arange(41),'fast',True)==0)
    rates=q.physical_realization(.2)
    z=two_state_step(rates['z0'],rates['background_oxidation_per_min'],rates['exchange_rest_per_min'],100.)
    assert z==pytest.approx(rates['z0']) # baseline, no added ligand


def test_pulse_current_and_weight_envelope():
    t=np.linspace(0,400,201)
    lo,hi=current_change_bounds(t,400,350,(5,40),(8,70))
    for w in np.linspace(0,1,11):
        d=a_current(t,350,8,70,w)-a_current(t,400,5,40,w)
        assert np.all(d>=lo-1e-12) and np.all(d<=hi+1e-12)
    assert lo[0]==hi[0]==-50
    lo,hi=current_change_bounds(t,400,400,(5,40),(8,70))
    assert np.all(lo>=0) and np.all(hi>=0)
    assert np.all(a_current(t,0,5,40,.5)==0)
    independent_lo,independent_hi=current_change_bounds(t,400,400,(5,40),(8,70),shared_weight=False)
    assert np.all(independent_lo<=lo+1e-12) and np.all(independent_hi>=hi-1e-12)
    assert independent_lo[10]<0<independent_hi[10]


def test_voltage_law_and_native_gain_guard():
    b=voltage_exchange(np.array([-100,0,100]),.01,.2,-30,10)
    assert np.all(np.diff(b)>0) and np.all((b>=.01)&(b<=.21))
    with pytest.raises(ValueError):require_branch_gain(None,'dFB','dFB')
    with pytest.raises(ValueError):require_branch_gain(1,'l-LNv','dFB')
    assert require_branch_gain(1,'dFB','dFB')==1


def test_invalid_modes_times_and_current_units_rejected():
    p=EffectiveMemory(.1,.1,.2,.3,.7)
    with pytest.raises(ValueError):p.coordinate([-1])
    with pytest.raises(ValueError):p.log_tau_ratio([1],'nuclear')
    with pytest.raises(ValueError):a_current([0],1,0,1,.5)
    with pytest.raises(ValueError):a_current([0],1,1,1,np.nan)
    with pytest.raises(ValueError):EffectiveMemory(np.nan,.1,.2,.3,.7)
