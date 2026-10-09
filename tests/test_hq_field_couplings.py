import numpy as np
from scipy.spatial.transform import Rotation
from impl.hq_field_couplings import well,dissipator_sum,BoundFieldModel
from impl.hq_rotation import dense_dissipator,frozen
from impl.hq_binding import wobble_operators


def test_well_no_field_and_detailed_balance_limit():
 w=well([.3,.4,np.sqrt(.75)],5,1e-11,0)
 np.testing.assert_allclose(w['rotvec'],0,atol=1e-15)
 np.testing.assert_allclose(w['K'],3/np.deg2rad(5)**2*np.eye(3),atol=1e-12)
 np.testing.assert_allclose(w['taus'],1e-11,rtol=1e-14)
 assert abs(w['rms_deg']-5)<1e-12 and abs(w['free_energy_residual'])<1e-14


def test_well_curvature_matches_independent_finite_difference():
 n=np.array([.6,0,.8]);w=well(n,5,1e-11,10.910941626120245)
 def energy(x):
  R=Rotation.from_rotvec(x).as_matrix()@w['R'];rv=Rotation.from_matrix(R).as_rotvec()
  return .5*w['c']*(rv@rv)-10.910941626120245*(n@R[:,2])
 h=1e-4;z=np.zeros(3);fd=np.zeros((3,3));grad=[]
 for i in range(3):
  ei=np.eye(3)[i]*h;grad.append((energy(ei)-energy(-ei))/(2*h))
  fd[i,i]=(energy(ei)+energy(-ei)-2*energy(z))/h**2
  for j in range(i):
   ej=np.eye(3)[j]*h;fd[i,j]=fd[j,i]=(energy(ei+ej)+energy(-ei-ej)-energy(ei-ej)-energy(-ei+ej))/(4*h*h)
 np.testing.assert_allclose(grad,0,atol=3e-8)
 np.testing.assert_allclose(fd,w['K'],rtol=2e-7,atol=1e-6)


def test_batched_lindblad_matches_independent_kron():
 rng=np.random.default_rng(92026);a=rng.normal(size=(5,3,3))+1j*rng.normal(size=(5,3,3))
 expected=sum(dense_dissipator(v) for v in a)
 np.testing.assert_allclose(dissipator_sum(a),expected,atol=1e-12)


def test_bound_zero_electric_recovers_existing_hq_wobble():
 f=frozen();responses={s:np.zeros((3,len(h),len(h)),complex) for s,h in f['body'].items()}
 model=BoundFieldModel(f,responses,0.,0.,1.)
 current=model.at('H',np.array([0.,0.,1.]));old,_=wobble_operators(f,1.,1e-11)
 np.testing.assert_allclose(current['body'],old['H']['body'],rtol=1e-12,atol=1e-7)
 np.testing.assert_allclose(current['D'],old['H']['D'],rtol=2e-7,atol=2e-9)


def test_field_wobble_preserves_spin_identity():
 f=frozen();rng=np.random.default_rng(100951)
 response={}
 for st,h in f['body'].items():
  a=rng.normal(size=(3,len(h),len(h)))+1j*rng.normal(size=(3,len(h),len(h)))
  response[st]=1e4*(a+a.conj().transpose(0,2,1))
 model=BoundFieldModel(f,response,14.,10.910941626120245,5.)
 for st in ('H','E'):
  current=model.at(st,np.array([.3,.4,np.sqrt(.75)]))
  assert model.validity[st]['max_unital_defect']<1e-9
  np.testing.assert_allclose(current['body'],current['body'].conj().T,atol=1e-10)
