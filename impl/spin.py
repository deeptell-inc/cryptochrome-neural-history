"""Two-electron/two-nucleus reaction instrument in arbitrary reaction units.

Observable: unconditional singlet product yield. Nuclear recycling, exchange
encoding and retention are hypotheses, not measured properties of brain CRY.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve

PAULI = [np.eye(2), np.array([[0, 1], [1, 0]]),
         np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
BASIS = np.array([np.kron(a, b)/2 for a in PAULI for b in PAULI], complex)
B = np.column_stack([a.reshape(-1, order='F') for a in BASIS])
X0 = np.zeros(16); X0[0] = .5  # sigma = I_4/4; B_0=I_4/2
singlet = np.array([0, 1, -1, 0])/np.sqrt(2)
SE = np.outer(singlet, singlet)
PS = np.kron(SE, np.eye(4))
PT = np.eye(16)-PS


def spin_operator(site, axis):
    factors = [np.eye(2, dtype=complex) for _ in range(4)]
    factors[site] = PAULI[axis+1]/2
    out = factors[0]
    for factor in factors[1:]:
        out = np.kron(out, factor)
    return out


OPS = [[spin_operator(site, a) for a in range(3)] for site in range(4)]


@dataclass(frozen=True)
class SpinParameters:
    field: float = .35
    exchange_offset: float = .4
    exchange_gain: float = 1.2
    k_s: float = 1.
    k_t: float = .4
    electron_dephasing: float = 0.
    hyperfine_scale: float = 1.


def hamiltonian(u, p=SpinParameters()):
    if not np.isfinite(u) or abs(u) > 1:
        raise ValueError('input must lie in [-1,1]')
    h = p.field*(OPS[0][2]+1.04*OPS[1][2])
    # Arbitrary anisotropic tensors, never attributed to a CRY species.
    a1 = np.array([[1.4, .15, 0], [.15, .7, .1], [0, .1, .3]])
    a2 = np.array([[.55, 0, .12], [0, 1.1, 0], [.12, 0, .4]])
    for a in range(3):
        for b in range(3):
            h = h+p.hyperfine_scale*(a1[a,b]*OPS[0][a]@OPS[2][b]+a2[a,b]*OPS[1][a]@OPS[3][b])
    j = p.exchange_offset+p.exchange_gain*u
    return h+j*sum(OPS[0][a]@OPS[1][a] for a in range(3))


def liouvillian(u, p=SpinParameters()):
    if p.k_s <= 0 or p.k_t <= 0 or p.electron_dephasing < 0:
        raise ValueError('positive reaction rates and nonnegative dephasing required')
    h = hamiltonian(u, p)
    k = p.k_s*PS+p.k_t*PT
    eye = np.eye(16)
    l = -1j*(np.kron(eye,h)-np.kron(h.T,eye))
    l -= .5*(np.kron(eye,k)+np.kron(k.T,eye))
    for site in [0,1]:
        a = OPS[site][2]
        # 2 gamma D[Sz] gives gamma for a single electron off-diagonal.
        l += 2*p.electron_dephasing*(np.kron(a.conj(),a)-.5*np.kron(eye,a@a)-.5*np.kron((a@a).T,eye))
    return l


def partial_trace_e(rho):
    return np.einsum('aiaj->ij',rho.reshape(4,4,4,4))


def instrument(u, p=SpinParameters()):
    injected = np.column_stack([np.kron(SE,a).reshape(-1,order='F') for a in BASIS])
    integrated = solve(-liouvillian(u,p),injected,assume_a='gen')
    branches=[]
    for rate,projector in [(p.k_s,PS),(p.k_t,PT)]:
        columns=[]
        for col in integrated.T:
            nuclear=rate*partial_trace_e(projector@col.reshape(16,16,order='F')@projector)
            columns.append(B.conj().T@nuclear.reshape(-1,order='F'))
        m=np.column_stack(columns)
        if np.max(np.abs(m.imag)) > 1e-10:
            raise ArithmeticError('Hermiticity not preserved')
        branches.append(m.real)
    return tuple(branches)


def choi(real_map):
    superop = B@real_map@B.conj().T
    # Blocks Phi(|i><j|); each block is an output operator.
    out=np.zeros((16,16),complex)
    for i in range(4):
        for j in range(4):
            unit=np.zeros((4,4));unit[i,j]=1
            block=(superop@unit.reshape(-1,order='F')).reshape(4,4,order='F')
            out[i*4:(i+1)*4,j*4:(j+1)*4]=block
    return out


@dataclass
class MapBank:
    alphabet: np.ndarray
    singlet: np.ndarray
    triplet: np.ndarray

    @classmethod
    def build(cls, alphabet=None, parameters=SpinParameters()):
        alphabet=np.linspace(-1,1,9) if alphabet is None else np.asarray(alphabet)
        branches=[instrument(float(u),parameters) for u in alphabet]
        return cls(alphabet,np.array([v[0] for v in branches]),np.array([v[1] for v in branches]))

    def transitions(self, retention=0., population_only=False):
        if not 0<=retention<=1:
            raise ValueError('retention must lie in [0,1]')
        m=self.singlet+self.triplet
        reset=np.zeros((16,16));reset[:,0]=2*X0
        m=retention*m+(1-retention)*reset
        if population_only:
            # Computational-basis nuclear dephasing; Pauli I/Z strings survive.
            keep=np.zeros(16);keep[[0,3,12,15]]=1
            m=m*keep[None,:,None]
        return m

    @property
    def effects(self):
        return 2*self.singlet[:,0,:]

    def sequence(self, indices, retention=.8, population_only=False):
        m=self.transitions(retention,population_only)
        state=X0.copy();y=np.empty(len(indices))
        for t,k in enumerate(indices):
            y[t]=self.effects[k]@state
            state=m[k]@state
        return y

    def batch(self, indices, retention=.8, population_only=False):
        """Independent trials, shape (time, trial); no shared molecular states."""
        m=self.transitions(retention,population_only);e=self.effects
        states=np.broadcast_to(X0,(indices.shape[1],16)).copy()
        y=np.empty(indices.shape)
        for t,k in enumerate(indices):
            y[t]=np.einsum('ni,ni->n',e[k],states)
            states=np.einsum('nij,nj->ni',m[k],states)
        return y


def chemical_filter(y,tau,dt,initial=None):
    if tau<0 or dt<=0:
        raise ValueError('tau >= 0 and dt > 0 required')
    if tau==0:
        return y.copy()
    a=np.exp(-dt/tau)
    state=np.zeros(y.shape[1:]) if initial is None else np.asarray(initial).copy()
    z=np.empty_like(y,dtype=float)
    for t,value in enumerate(y):
        state=a*state+(1-a)*value
        z[t]=state
    return z
