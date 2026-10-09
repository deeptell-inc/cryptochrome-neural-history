import numpy as np
from scipy import sparse as sp
from scipy.linalg import expm
from impl.hq_rotation import angular_basis,generator,ad,s
from impl.hq_electric import electric_basis,electric_generator


def toy():
    h=3e6*s.S[0]+7e6*s.S[2]
    return dict(body={'H':h},D={'H':np.zeros((4,4),complex)},spin={'H':s.S},Z={'H':s.S})


def test_boltzmann_moments_and_equilibrium_exact():
    for k in (0.,.109109416261,1.091094162612,10.91094162612):
        a=electric_basis(3,k);z=a['n'][:,2];w=a['weights']
        L=0 if not k else 1/np.tanh(k)-1/k
        np.testing.assert_allclose(w@z,L,atol=2e-13)
        np.testing.assert_allclose(a['scalar_rotation'][:,0],0,atol=1e-12)
        assert np.linalg.eigvalsh(-a['scalar_rotation']).min()>-1e-11


def test_zero_field_potential_recovers_original_diffusion():
    f=toy();a=electric_basis(2,0.);old=angular_basis(2)
    np.testing.assert_allclose(electric_generator(f,a,'H',2e-8).toarray(),generator(f,old,'H',2e-8).toarray(),rtol=1e-12,atol=1e-6)


def test_covariant_weak_form_matches_direct_node_gram_and_preserves_trace():
    f=toy();a=electric_basis(2,10.9,axis=0);d=2;na=len(a['lm']);rot=np.zeros((na*4,na*4),complex)
    for q,w in enumerate(a['weights']):
        for j in range(3):
            v=np.kron(a['derivatives'][j][q:q+1],np.eye(4))+np.kron(a['Y'][q:q+1],ad(f['spin']['H'][j]).toarray())
            rot+=w*np.einsum("ji,jk->ik",v.conj(),v,optimize=False)
    actual=electric_generator(f,a,'H',1/6,field_T=0)-electric_generator(f,a,'H',np.inf,field_T=0)
    np.testing.assert_allclose(actual.toarray(),-rot,atol=2e-9)
    ident=np.zeros(na*4,complex);ident[:4]=s.trvec(2)
    np.testing.assert_allclose(np.einsum("ij,j->i",rot,ident),0,atol=1e-11)
    np.testing.assert_allclose(np.einsum("i,ij->j",ident,rot),0,atol=1e-11)
    assert np.linalg.eigvalsh(rot).min()>-1e-10


def test_single_dipole_does_not_remove_axial_spin_rotation():
    a=electric_basis(2,10.9);f=toy()
    r=(electric_generator(f,a,'H',1/6,0)-electric_generator(f,a,'H',np.inf,0)).toarray()
    # Spin-only constant functions have all three spin-rotation contributions.
    expected=-sum(ad(j).toarray()@ad(j).toarray() for j in s.S)
    np.testing.assert_allclose(r[:4,:4],expected,atol=1e-10)


def test_scalar_drift_has_correct_sign_against_smoluchowski_equation():
    # Backward generator on f(z)=z: D_R[(1-z^2)f''+(-2z+k(1-z^2))f'].
    # Galerkin projection must match even though the degree-2 RHS is truncated at L=1.
    k=3.;a=electric_basis(3,k);z=a['n'][:,2]
    coeff=np.einsum('aq,q->a',a['inverse'],z)
    got=np.einsum('ab,b->a',a['scalar_rotation'],coeff)
    expected=np.einsum('aq,q->a',a['inverse'],-2*z+k*(1-z*z))
    np.testing.assert_allclose(got,expected,atol=3e-12)
