"""Conditional Hk redox memory and A-current observation models.

Chemical rates are /minute; channel inactivation and current use ms and pA.
Fitted x is an observable coordinate, NOT a measured NADP+ fraction or nuclear
population. No default CRY delivery, channel counts, or firing gain is supplied.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit


def two_state_step(z, oxidation_per_min, exchange_per_min, duration_min):
    """Exact population propagation for constant nonnegative chemical rates."""
    z = np.asarray(z, dtype=float)
    a, b, dt = oxidation_per_min, exchange_per_min, duration_min
    if not np.all(np.isfinite(z)) or np.any((z < 0) | (z > 1)):
        raise ValueError('redox fraction must lie in [0,1]')
    if not np.all(np.isfinite([a, b, dt])) or min(a, b, dt) < 0:
        raise ValueError('invalid rate or duration')
    rate = a + b
    return z.copy() if rate == 0 else z + (a / rate - z) * (-np.expm1(-rate * dt))


def voltage_exchange(V_mV, basal_per_min, amplitude_per_min, half_mV, slope_mV):
    """Candidate voltage law; all parameters require explicit calibration."""
    if not np.all(np.isfinite([basal_per_min, amplitude_per_min, half_mV, slope_mV])):
        raise ValueError('finite voltage-law parameters required')
    if min(basal_per_min, amplitude_per_min) < 0 or slope_mV <= 0:
        raise ValueError('invalid voltage law')
    if not np.all(np.isfinite(V_mV)):
        raise ValueError('invalid voltage')
    return basal_per_min + amplitude_per_min * expit((np.asarray(V_mV) - half_mV) / slope_mV)


@dataclass(frozen=True)
class EffectiveMemory:
    k_rest_per_min: float
    extra_train_per_min: float
    baseline_over_excursion: float
    log_tau_fast_excursion: float
    log_tau_slow_excursion: float

    def __post_init__(self):
        vals = list(self.__dict__.values())
        if not np.all(np.isfinite(vals)) or min(vals) < 0 or self.k_rest_per_min <= 0:
            raise ValueError('invalid effective parameters')

    def coordinate(self, times_min, train=False):
        """Constant 4-ONE exposure; optional 10–30 min voltage train.

        The fitted extra rate averages the full 10 Hz protocol, NOT a rate at
        +10 mV. Diagnostic clamp pulses at other times are folded into k_rest.
        """
        t = np.asarray(times_min, dtype=float)
        if np.any(t < 0) or not np.all(np.isfinite(t)):
            raise ValueError('times must be nonnegative and finite')
        k, v, B = self.k_rest_per_min, self.extra_train_per_min, self.baseline_over_excursion
        if not train:
            return -np.expm1(-k*t)
        x10 = -np.expm1(-10*k)
        target = (k-v*B)/(k+v)
        x30 = target+(x10-target)*np.exp(-20*(k+v))
        return np.where(t <= 10, -np.expm1(-k*t),
                        np.where(t <= 30, target+(x10-target)*np.exp(-(k+v)*np.maximum(t-10, 0)),
                                 1+(x30-1)*np.exp(-k*np.maximum(t-30, 0))))

    def log_tau_ratio(self, times_min, mode, train=False):
        if mode not in ('fast', 'slow'):
            raise ValueError('mode must be fast or slow')
        A = self.log_tau_fast_excursion if mode == 'fast' else self.log_tau_slow_excursion
        return A*self.coordinate(times_min, train)

    def physical_realization(self, excursion):
        """An arbitrary member of an EXACTLY indistinguishable redox family.

        D=excursion, z0=B*D, zrest=(B+1)*D. Observed log tau response =
        (A/D)*(z-z0). D cannot be inferred from these tau measurements.
        """
        B, k = self.baseline_over_excursion, self.k_rest_per_min
        if not np.isfinite(excursion) or not 0 < excursion <= 1/(1+B):
            raise ValueError('unphysical occupation scale')
        z0 = B*excursion
        a = k*(B+1)*excursion
        exchange = k-a
        background = exchange*z0/(1-z0)
        return dict(z0=z0, oxidation_per_min=a, exchange_rest_per_min=exchange,
                    extra_train_per_min=self.extra_train_per_min,
                    background_oxidation_per_min=background,
                    added_oxidation_per_min=a-background,
                    log_tau_fast_per_fraction=self.log_tau_fast_excursion/excursion,
                    log_tau_slow_per_fraction=self.log_tau_slow_excursion/excursion,
                    status='conditional_realization_not_native_estimate')


def a_current(time_ms, peak_pA, tau_fast_ms, tau_slow_ms, fast_weight):
    """Post-peak clamp current candidate, not a full voltage-gated neuron.

    Positive outward. No persistent component; its absence is an assumption.
    Mixture weight is unmeasured and MUST be supplied.
    """
    if not np.all(np.isfinite([peak_pA, tau_fast_ms, tau_slow_ms, fast_weight])):
        raise ValueError('finite current parameters required')
    if peak_pA < 0 or min(tau_fast_ms, tau_slow_ms) <= 0 or not 0 <= fast_weight <= 1:
        raise ValueError('invalid current parameters')
    t = np.asarray(time_ms, dtype=float)
    if np.any(t < 0) or not np.all(np.isfinite(t)):
        raise ValueError('post-peak time must be nonnegative')
    return peak_pA*(fast_weight*np.exp(-t/tau_fast_ms)+(1-fast_weight)*np.exp(-t/tau_slow_ms))


def current_change_bounds(time_ms, peak_before, peak_after, taus_before, taus_after, *, shared_weight=True):
    """Exact envelope over common or independently changing mixture weights.

    Invariance of the weight under redox perturbation is not established by
    the source summary table. Both choices must be exposed in applications.
    """
    pairs = [(0.,0.),(1.,1.)] if shared_weight else [(a,b) for a in (0.,1.) for b in (0.,1.)]
    differences = np.array([a_current(time_ms, peak_after, *taus_after, after)
                            -a_current(time_ms, peak_before, *taus_before, before) for before,after in pairs])
    return differences.min(axis=0), differences.max(axis=0)


def require_branch_gain(branch_to_local_uM, cell_system, calibrated_cell_system):
    """Prevent the dFB partial fit being mistaken for an end-to-end CRY gain."""
    if branch_to_local_uM is None:
        raise ValueError('CRY branch-to-local-concentration gain is unidentified')
    if not np.isfinite(branch_to_local_uM) or branch_to_local_uM < 0:
        raise ValueError('invalid branch gain')
    if cell_system != calibrated_cell_system:
        raise ValueError('cell-system transfer requires an independent calibration')
    return branch_to_local_uM
