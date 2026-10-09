import numpy as np
from impl.hq_binding import wobble_operators
from impl.hq_rotation import angular_basis,s


def toy():
    h=2e7*s.S[2]
    return dict(body={k:h for k in ('H','R','E','RP')},
                D={k:np.zeros((4,4),complex) for k in ('H','R','E','RP')},
                spin={k:s.S for k in ('H','R','E','RP')})


def test_zero_wobble_recovers_original():
    f=toy();o,_=wobble_operators(f,0.,1e-11)
    for k in o:
        np.testing.assert_array_equal(o[k]['body'],f['body'][k])
        np.testing.assert_array_equal(o[k]['D'],f['D'][k])


def test_white_noise_longitudinal_rate_has_correct_factor():
    f=toy();angle=.02;tau=1e-11;o,_=wobble_operators(f,np.rad2deg(angle),tau)
    z=s.vec(s.S[2]);expected=-2*(angle**2/3)*tau*(2e7)**2*z
    np.testing.assert_allclose(o['H']['extra']@z,expected,rtol=2e-14,atol=1e-12)


def test_noise_is_trace_preserving_and_unital():
    f=toy();o,_=wobble_operators(f,5.,1e-11)
    tr=s.trvec(2)
    np.testing.assert_allclose(tr@o['H']['D'],0.,atol=1e-10)
    np.testing.assert_allclose(o['H']['D']@tr,0.,atol=1e-10)


def test_capture_profile_and_isotropic_free_pool_are_stationary():
    a=angular_basis(3,quadrature_extra=-1)
    for bias in (-.9,0,1.8):
        profile=1+bias*(3*a['n'][:,2]**2-1)/2
        assert profile.min()>0
        np.testing.assert_allclose(a['weights']@profile,1.,atol=1e-14)
        f=.5;off=100.;on=f/(1-f)*off
        free=np.zeros(len(a['lm']),complex);free[0]=1-f
        bound=f*profile
        jump=on*profile*np.einsum('qa,a->q',a['Y'],free)-off*bound
        np.testing.assert_allclose(jump,0.,atol=1e-12)
