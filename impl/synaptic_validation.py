"""Items 2--5: conditional identifiability and acquisition tests, not native data.

Reuses the frozen sparse circuit. Synthetic edge IDs never map to connectome IDs.
"""
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t, norm
from . import sparse_synaptic as s

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/synaptic-validation'


def impulse_bank(times, delays, leak=.6):
    age = np.asarray(times)[None, :] - np.asarray(delays)[:, None]
    return np.where(age >= -1e-14, np.exp(-leak * np.maximum(age, 0)), 0.)


def fit_impulses(traces, times, delay_grid):
    """Fit an amplitude and delay to each edge's mean trace; no test traces used."""
    h = impulse_bank(times, delay_grid)
    projection = np.asarray(traces) @ h.T
    improvement = projection**2 / (h*h).sum(1)
    index = np.argmax(improvement, axis=-1)
    amplitude = np.take_along_axis(projection, index[..., None], axis=-1)[..., 0] / (h*h).sum(1)[index]
    return amplitude, np.asarray(delay_grid)[index]


def calibration_experiment(seed, noise_scale, n_train=16, n_valid=16):
    rng = np.random.default_rng(seed)
    w, target = s.network()
    times = np.arange(251)*.001
    delays = np.random.default_rng(260916401).integers(2, 31, s.N)*.001
    true_amp = w*(target == 0)
    h = impulse_bank(times, delays)
    window = (times >= .02) & (times <= .20)
    true_score = (true_amp[:, None]*h)[:, window].mean(1)
    noise_sd = noise_scale*np.median(abs(true_amp[true_amp != 0]))
    def observations(n):
        # A cluster is one synthetic specimen; trials/time points never become n.
        amp = true_amp[None, :]*(1+.15*rng.normal(size=(n, s.N)))
        amp += .5*noise_sd*rng.normal(size=(n, s.N))
        eps = rng.normal(size=(n, s.N, len(times)))*noise_sd
        for k in range(1, len(times)):
            eps[:, :, k] = .8*eps[:, :, k-1]+.6*eps[:, :, k]
        return amp[:, :, None]*h[None, :, :]+eps
    train, valid = observations(n_train), observations(n_valid)
    a, delay = fit_impulses(train.mean(0), times, np.arange(41)*.001)
    fitted = impulse_bank(times, delay)
    scale = fitted[:, window].mean(1)
    scores = a*scale
    selected = np.argsort(-scores)[:s.K]
    # Hold delay fixed, estimate per-specimen validation influence; no reselection.
    valid_amp = (valid*fitted[None, :, :]).sum(2)/(fitted*fitted).sum(1)
    valid_scores = valid_amp*scale
    mean = valid_scores.mean(0)
    sem = valid_scores.std(0, ddof=1)/np.sqrt(n_valid)
    critical = student_t.ppf(1-.05/(2*s.N), n_valid-1)
    low, high = mean-critical*sem, mean+critical*sem
    _, valid_delay = fit_impulses(valid.mean(0), times, np.arange(41)*.001)
    true_top = np.argsort(-true_score)[:s.K]
    row = dict(seed=seed, noise_scale=noise_scale, noise_sd_state=noise_sd,
               n_train_clusters=n_train, n_validation_clusters=n_valid,
               top8_recall=len(set(selected)&set(true_top))/s.K,
               selected_true_influence=true_score[selected].sum(),
               oracle_influence=true_score[true_top].sum(),
               retained_influence=true_score[selected].sum()/true_score[true_top].sum(),
               validation_positive_simultaneous=int(np.sum(low[selected] > 0)),
               validation_mean_influence=mean[selected].sum(),
               selected_delay_MAE_ms=float(abs(delay[selected]-delays[selected]).mean()*1000),
               validation_delay_MAE_ms=float(abs(valid_delay[selected]-delays[selected]).mean()*1000),
               native_measurement=False)
    edges = [dict(seed=seed, noise_scale=noise_scale, n_train_clusters=n_train,
                  synthetic_edge=f'SYN{j:02d}', selected=bool(j in selected),
                  true_influence=true_score[j], training_influence=scores[j],
                  validation_influence=mean[j], validation_SEM=sem[j],
                  validation_simultaneous_low=low[j], validation_simultaneous_high=high[j],
                  true_delay_s=delays[j], training_delay_s=delay[j],
                  validation_delay_s=valid_delay[j], native_id=None) for j in range(s.N)]
    masks = {'aligned': true_top, 'random': np.random.default_rng(991).choice(s.N,s.K,replace=False),
             'zero_P_influence': np.flatnonzero(target == 1)[:s.K],
             'opposite': np.argsort(true_score)[:s.K]}
    overlap = []
    for name, ids in masks.items():
        mask = np.isin(np.arange(s.N), ids)
        # Fixed per-selected-edge exposure: missing overlap is not renormalized away.
        effective = mask*np.isin(np.arange(s.N), selected)/s.K
        overlap.append(dict(seed=seed, noise_scale=noise_scale, n_train_clusters=n_train,
                            mask=name, overlap_count=int(mask[selected].sum()),
                            delivered_budget=effective.sum(),
                            signed_preparation_transfer=float(true_score@effective),
                            native_measurement=False))
    return row, edges, overlap


def block_observation(baseline_shift=0., source_ratio=1., pathway_ratio=1., direct_ratio=1., removed=.8):
    """Single edge local linear model; molecule on/off x pre/post block.

    pathway_ratio is independently measured with classical input to that path,
    not inferred from a direct electrode pulse at the target.
    """
    before_off, before_on = 1., 1.1
    after_off = 1.+baseline_shift
    after_on = after_off+.1*source_ratio*pathway_ratio*(1-removed)
    ratio = (after_on-after_off)/(before_on-before_off)
    corrected = None if source_ratio*pathway_ratio == 0 else 1-ratio/(source_ratio*pathway_ratio)
    return dict(before_off=before_off, before_on=before_on, after_off=after_off, after_on=after_on,
                source_ratio=source_ratio, pathway_ratio=pathway_ratio, direct_ratio=direct_ratio,
                true_removed=removed, naive_removed=1-ratio,
                source_and_direct_corrected=None if source_ratio*direct_ratio == 0 else 1-ratio/(source_ratio*direct_ratio),
                source_and_pathway_corrected=corrected,
                raw_on_change=after_on-before_on,
                DID=(after_on-after_off)-(before_on-before_off), native_measurement=False)


def equivalence_interval(log_ratio, sem, margin=.10, comparisons=3):
    """Conservative simultaneous CI within predeclared +/- margin equivalence bounds."""
    z = norm.ppf(1-.05/(2*comparisons))
    lower, upper = np.exp(log_ratio-z*sem), np.exp(log_ratio+z*sem)
    return lower, upper, bool(lower >= 1-margin and upper <= 1+margin)


def nuisance_residual(signal, design):
    signal = np.asarray(signal, float)
    design = np.asarray(design, float)
    coef = np.linalg.lstsq(design, signal, rcond=None)[0]
    return signal-design@coef, int(np.linalg.matrix_rank(design))


def release_bound(delta_yield, tau=.012, gain=512., initial_age=.007, horizon=.8):
    """Union bound on differing releases under shared-uniform coupling.

    Also bounds any binary downstream event TV distance, for identical backgrounds.
    Allocation nonnegative and total <=1. Not an energetic or native bound.
    """
    ages = initial_age+np.arange(0., horizon, s.PERIOD)
    return min(1., gain*abs(delta_yield)*np.exp(-ages/tau).sum())


def simulate_stop_clock(detector_rt, delta_yield, gain=512., tau=.012, seed=26093691, dt=.002):
    """Compare stop-activation-born versus cue-born product; local release clock.

    Stop variable is inactive until detection. Cue signal ages through detector
    latency; no reservoir loading before stop activation is allowed in this arm.
    Detection failures do not get a stop intervention. This is one mechanistic
    hypothesis, not a universal limit on upstream preloading/chemical storage.
    """
    if dt not in (.001,.002):
        raise ValueError('supported time grids are 1 and 2ms')
    d = np.asarray(detector_rt)
    n = len(d); horizon=.8
    w, target = s.network()
    allocation, ids = s.allocations(s.influence(w, target, 'stop'))
    alpha = allocation[s.ARMS.index('targeted')]
    rng=np.random.default_rng(seed); syn=np.random.default_rng(seed+100000)
    x=np.zeros((3,n)); hits=np.full((3,n),-1,int)
    expected=np.zeros((3,n)); log_equal=np.zeros((3,n))
    for k in range(round(horizon/dt)):
        time=k*dt
        if k % round(s.PERIOD/dt) == 0:
            event=np.full(n, 0. if time < .003 else delta_yield*gain*np.exp(-(time-.003)/tau))
            age=time+d-.003
            cue=np.where((age>=0)&np.isfinite(d),delta_yield*gain*np.exp(-np.maximum(age,0)/tau),0.)
            event[~np.isfinite(d)]=0.
            pulses=np.stack([np.zeros(n),event,cue])
            u=syn.random((n,s.N))
            base=u<s.P0
            x += ((base.astype(float)-s.P0)@w)[None,:]
            for arm in (1,2):
                dp=pulses[arm,:,None]*alpha[None,:]
                if np.any(s.P0+dp>1) or np.any(s.P0+dp<0):
                    raise ValueError('invalid release probability')
                release=u<(s.P0+dp)
                x[arm] += ((release.astype(int)-base)@w)
                expected[arm] += dp.sum(1)
                log_equal[arm] += np.log1p(-abs(dp)).sum(1)
        eta=np.zeros(n)
        for _ in range(round(dt/.001)):
            eta += rng.normal(size=n)*.65*np.sqrt(.001)
        x += 5*dt+eta[None,:]
        crossed=(hits<0)&(x>=1);hits[crossed]=k
    rt=np.where(hits>=0,(hits+1)*dt,np.inf)
    records=[]
    for arm,name in enumerate(('baseline','activation_born','cue_born')):
        diff=np.minimum(rt[arm],horizon)-np.minimum(rt[0],horizon)
        records.append(dict(arm=name, seed=seed, n=n, dt_s=dt, tau_s=tau,
                            delta_RMST_ms=1000*diff.mean(), MC_SE_ms=1000*diff.std(ddof=1)/np.sqrt(n),
                            expected_extra_release=expected[arm].mean(),
                            binary_event_TV_upper=float((-np.expm1(log_equal[arm])).mean()),
                            native_prediction=False))
    return dict(rt=rt,records=records,expected=expected,detector_rt=d)
