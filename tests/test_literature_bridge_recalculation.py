import numpy as np
from scipy.linalg import expm
from impl import literature_bridge_recalculation as r
from impl.chemical_carrier_mapping import cofactor_generator

def test_independent_implicit_solution_and_probability_tangent():
    a=r.solve_pulse(10,1,100,.05);b=r.solve_pulse(10,1,100,.05,method='Radau')
    t=np.r_[0,np.geomspace(1e-5,10,120)]
    np.testing.assert_allclose(a.sol(t),b.sol(t),atol=2e-9)
    p=a.sol(t)[:3];s=a.sol(t)[3:]
    assert p.min()>-1e-10
    np.testing.assert_allclose(p.sum(0),1,atol=1e-10)
    np.testing.assert_allclose(s.sum(0),0,atol=1e-10)

def test_tangent_matches_finite_difference():
    eps=1e-4;t=[.04,.11,.2,.4,2]
    sol=r.solve_pulse(10,.01,100,.2)
    plus=r.solve_pulse(10*np.exp(eps),.01,100,.2)
    minus=r.solve_pulse(10*np.exp(-eps),.01,100,.2)
    np.testing.assert_allclose(sol.sol(t)[3:],(plus.sol(t)[:3]-minus.sol(t)[:3])/(2*eps),atol=1e-8)

def test_stationary_and_no_write_retention():
    G=cofactor_generator(10,1,100,0)
    st=expm(G*50)@np.array([1.,0.,0.])
    assert np.isclose(st[1],r.held_steady_occupancy(10,1,100))
    for off in [.01,1,100]:
        G=cofactor_generator(0,off,100,0)
        out=expm(G*.2)@np.array([0.,1.,0.])
        assert np.isclose(out[1],np.exp(-off*.2))

def test_peak_and_old_special_case():
    sol=r.solve_pulse(10,.01,100,.2);t,p=r.peak(sol)
    assert 0<t<10 and p>=np.max(sol.sol(np.linspace(0,10,1000))[1])-1e-8
    np.testing.assert_allclose(r.old_pulse([0,1],1,1),[0,np.exp(-1)])
