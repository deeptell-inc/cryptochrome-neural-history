"""Conditional P/E chemistry -> saturating gate -> bounded K current.

No native gain is inferred. P/E label branch-derived signals, not identified
ligands. A is P-signal concentration / its effective half-saturation scale;
eta_E includes E/P formation, access and potency. Reference pools are maintained
by identical external sources in both arms; only their initial differences decay.
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

AVOGADRO = 6.02214076e23


def source_concentration_uM(molecules, reacting_fraction, delivery, volume_pL):
    if molecules < 0 or not 0 <= reacting_fraction <= 1 or not 0 <= delivery <= 1 or volume_pL <= 0:
        raise ValueError('invalid molecular source')
    return molecules * reacting_fraction * delivery / (AVOGADRO * volume_pL * 1e-12) * 1e6


def gate(x):
    return x / (1 + x)


@dataclass(frozen=True)
class Scenario:
    A: float = 2.5
    eta_E: float = 0.
    tau_P_s: float = .012
    tau_E_s: float = .012
    tau_gate_s: float = 0.
    capacity_nS: float = .2
    gK_reference_nS: float = .2
    gL_nS: float = .8
    C_pF: float = 20.
    EL_mV: float = -40.
    EK_mV: float = -80.
    channel_response: str = 'linear_gate'

    @property
    def Vref_mV(self):
        return (self.gL_nS*self.EL_mV + self.gK_reference_nS*self.EK_mV)/(self.gL_nS+self.gK_reference_nS)


def baseline(y0, p):
    return gate(p.A*(y0+p.eta_E*(1-y0)))


def max_capacity(y0, p):
    """Ensures gK >= 0 for every possible 0 <= h <= 1, not just this trace."""
    return p.gK_reference_nS/(1-baseline(y0, p))


def signal(t, y0, dy, p):
    t=np.asarray(t)
    P=p.A*(y0+dy*np.exp(-t/p.tau_P_s))
    E=p.A*((1-y0)-dy*np.exp(-t/p.tau_E_s))
    return P, E, P+p.eta_E*E


def equilibrium_gate_difference(t, y0, dy, p):
    # Stable difference, including exact equal-activity/equal-lifetime null.
    x0=p.A*(y0+p.eta_E*(1-y0))
    dx=p.A*dy*(np.exp(-np.asarray(t)/p.tau_P_s)-p.eta_E*np.exp(-np.asarray(t)/p.tau_E_s))
    return dx/((1+x0)*(1+x0+dx))


def simulate(y0, dy, p=Scenario(), times=None, method='DOP853', rtol=2e-9):
    if not 0 <= y0 <= 1 or not 0 <= y0+dy <= 1:
        raise ValueError('invalid yields')
    if min(p.A,p.eta_E,p.tau_gate_s,p.capacity_nS) < 0 or min(p.tau_P_s,p.tau_E_s,p.gK_reference_nS,p.gL_nS,p.C_pF) <= 0:
        raise ValueError('invalid scenario')
    if p.channel_response not in ('linear_gate','bounded_amplifier'):
        raise ValueError('unknown channel response')
    if p.channel_response=='linear_gate' and p.capacity_nS > max_capacity(y0,p)*(1+1e-12):
        raise ValueError('capacity permits negative K conductance')
    times=np.linspace(0,.2,801) if times is None else np.asarray(times)
    if times[0] != 0 or np.any(np.diff(times)<=0):
        raise ValueError('times must start at zero and increase')
    gt=p.gL_nS+p.gK_reference_nS
    def channel_change(dh):
        if p.channel_response=='linear_gate':return p.capacity_nS*dh
        # Counterexample: extra sensitive but bounded relay, NOT an inferred Hk law.
        # capacity is the LOCAL slope in this model, not a full-range capacity.
        return p.gK_reference_nS*np.tanh(p.capacity_nS/p.gK_reference_nS*dh)
    def rhs(t,z):
        target=equilibrium_gate_difference(t,y0,dy,p)
        dh=target if p.tau_gate_s==0 else z[1]
        dg=channel_change(dh)
        dv=1000/p.C_pF*(-(gt-dg)*z[0]+dg*(p.Vref_mV-p.EK_mV))
        return [dv,0. if p.tau_gate_s==0 else (target-z[1])/p.tau_gate_s]
    sol=solve_ivp(rhs,(0,times[-1]),[0.,0.],t_eval=times,method=method,rtol=rtol,atol=1e-12,max_step=.002)
    if not sol.success:raise RuntimeError(sol.message)
    dh=equilibrium_gate_difference(times,y0,dy,p) if p.tau_gate_s==0 else sol.y[1]
    h=baseline(y0,p)+dh;gK=p.gK_reference_nS-channel_change(dh)
    dI=channel_change(dh)*(p.Vref_mV-p.EK_mV) # inward-positive change at fixed Vref
    if np.min(h)<-1e-9 or np.max(h)>1+1e-9 or np.min(gK)<-1e-9:
        raise AssertionError('nonphysical gate or conductance')
    return dict(time_s=times,delta_V_mV=sol.y[0],delta_I_clamp_pA=dI,delta_h=dh,gK_nS=gK,h=h)


def metrics(trace):
    v=trace['delta_V_mV'];i=trace['delta_I_clamp_pA'];t=trace['time_s']
    j=int(np.argmax(abs(v)))
    return dict(peak_signed_V_mV=float(v[j]),peak_time_s=float(t[j]),
                max_V_mV=float(max(v)),min_V_mV=float(min(v)),
                max_I_pA=float(max(i)),min_I_pA=float(min(i)),
                I_initial_pA=float(i[0]),V_50ms_mV=float(np.interp(.05,t,v)))


def calibration_current(concentrations_nM, log_parameters, drive_mV=32.):
    """Independent zero-ligand clamp assay, inward-positive K suppression.

    Known absolute P/E signals and both pure-branch inputs are ideal assumptions.
    Parameters: available conductance, K_P, K_E. This assay does not itself
    calibrate CRY abundance/delivery or transfer to a different baseline.
    """
    capacity,KP,KE=np.exp(log_parameters)
    c=np.asarray(concentrations_nM)
    x=c[:,0]/KP+c[:,1]/KE
    return drive_mV*capacity*gate(x)
