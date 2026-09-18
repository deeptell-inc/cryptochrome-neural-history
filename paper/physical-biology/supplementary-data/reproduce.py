"""Regenerate article figures and check selected results from frozen inputs.

No upstream molecular-dynamics calculation or complete simulation rerun.
Run: python reproduce.py
Dependencies: numpy, scipy, pandas, matplotlib.
"""
from pathlib import Path
import hashlib
import json
import math
import os

HERE = Path(__file__).resolve().parent
OUT = HERE / 'reproduced'
OUT.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(OUT / '.mpl-cache'))
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from scipy.stats import binom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                     'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
BLUE, ORANGE, GREEN = '#0072B2', '#D55E00', '#009E73'


def csv(path):
    return pd.read_csv(HERE / path)


def save(fig, name):
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / (name + '.' + ext), bbox_inches='tight', dpi=180)
    plt.close(fig)


def recurrence(N, p):
    m = min(N, math.floor((N + 1) * p))
    a = np.zeros(N + 1); a[m] = 1
    for k in range(m, N):
        a[k+1] = a[k] * (N-k)/(k+1) * p/(1-p)
    for k in range(m, 0, -1):
        a[k-1] = a[k] * k/(N-k+1) * (1-p)/p
    return a/a.sum()


def main():
    hashes = json.loads((HERE / 'source-hashes.json').read_text())
    assert all(hashlib.sha256((HERE / f).read_bytes()).hexdigest() == h for f, h in hashes.items())
    m = csv('tables/population_modes_8.csv')
    z = np.load(HERE / 'data/population-memory/modes_and_states.npz')
    x0 = np.eye(9).reshape(-1, order='F') / 9
    trace = np.eye(9).reshape(-1, order='F')
    reset = np.outer(x0, trace)
    DP = np.einsum('ik,jk->ij', z['C'], z['C'].conj())
    M = z['cycle']
    operators = np.load(HERE / 'data/population-memory/latest_case.npz')
    read = np.einsum('i,ij->j', trace, operators['P_0'])
    # These archive entries are prepared states, not readout rows:
    # reaction_P_weighted_g0 = P_0 x0. Use trace(P_0 rho) above.
    updates = {'full': M,
               'population_HQ': np.einsum('ij,jk,kl->il', DP, M, DP),
               'reset': np.einsum('ij,jk,kl->il', reset, M, reset)}
    yields = {}
    for name, A in updates.items():
        x = x0.copy()
        for _ in range(20):
            x = np.einsum('ij,j->i', A, x)
        value = np.einsum('i,i->', read, x)
        assert abs(value.imag) < 1e-9 and np.isfinite(value)
        yields[name] = float(value.real)
    base = csv('data/population-memory/baseline_comparison.csv').set_index('model')
    for key, value in yields.items():
        assert abs(value - base.loc[key, 'yield_P']) < 3e-10
    src = csv('data/dark-basis-resolution/conditional_products_spikes.csv')
    src = src[(src.condition == 'upcj2N__full') & (src.gamma_oxygen_s == 0)].set_index('model')
    q0, q1 = float(src.loc['full', 'yield_reference']), float(src.loc['full', 'yield_P'])
    dq = q1-q0
    w = m.write_amplitude.to_numpy() * m.read_sensitivity.to_numpy()
    rates = m.decay_rate_s.to_numpy()
    half = brentq(lambda t: np.dot(w, np.exp(-rates*t)) - w.sum()/2, 0, 100)
    assert abs(w.sum()-dq) < 3e-10 and abs(half-1.63473966727) < 1e-8
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.5), layout='constrained')
    colors = [GREEN, BLUE, ORANGE, GREEN, BLUE, ORANGE, ORANGE, ORANGE]
    ax[0].bar(m['mode'], 100*w/w.sum(), color=colors)
    ax[0].set(xlabel='HQ population mode', ylabel='Readout contribution (%)', title='(a) Mode-selective readout', xticks=range(1,9))
    from matplotlib.patches import Patch
    ax[0].legend(handles=[Patch(color=BLUE,label='N5'),Patch(color=GREEN,label='N10'),Patch(color=ORANGE,label='Correlation')], fontsize=8)
    t = np.logspace(-3, 2, 220)
    hold = csv('data/population-memory/HQ_pulse_chase.csv')
    h = hold[hold.time_s >= 1e-3]
    ax[1].semilogx(h.time_s, h.full_yield_memory/dq, color=BLUE, label='Full')
    ax[1].semilogx(t, np.einsum('ij,j->i', np.exp(-t[:,None]*rates), w)/w.sum(), '--', color=ORANGE, label='Population modes')
    ax[1].axvline(half, ls=':', color='gray', label='Half-decay: 1.635 s')
    ax[1].set(xlabel='Isolated HQ hold (s)', ylabel='Normalised yield contrast', title='(b) Conditional storage')
    ax[1].legend(fontsize=8)
    save(fig, 'fig1')

    peaks = csv('data/literature-bridge-recalculation/peak_comparison.csv')
    peaks = peaks[(peaks.a0_s == 10) & (peaks.off_s == .01)].sort_values('tau_s')
    # Column names are preserved from original output; no fit is performed.
    peak_col = 'new_peak_oxidized_fraction'
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.5), layout='constrained')
    t = np.linspace(0, 5, 300)
    ax[0].plot(t, 100*dq/2*np.exp(-t), color=GREEN, lw=2)
    ax[0].set(xlabel='Time after reaction (s)', ylabel='100 × peroxide contrast / HQ event', title='(a) Equal chemical endpoints')
    ax[1].bar(np.arange(len(peaks)), 100*peaks[peak_col], color=ORANGE)
    ax[1].set(xticks=np.arange(len(peaks)), xticklabels=[f'{1000*t:g}' for t in peaks.tau_s], xlabel='Assumed input lifetime (ms)', ylabel='Peak oxidised fraction (%)', title='(b) Supplied Hk input')
    save(fig,'fig2')
    chemical = solve_ivp(lambda t,y: [-5*y[0]-dq/2*np.exp(-t)], [0,2], [dq/2], t_eval=[0,.2,2], rtol=1e-11, atol=1e-15)
    analytic = dq/2*(np.exp(-5*chemical.t)-(np.exp(-chemical.t)-np.exp(-5*chemical.t))/4)
    assert np.max(abs(analytic-chemical.y[0])) < 1e-12

    c = csv('tables/circuit_confirmations_6.csv')
    fig, ax = plt.subplots(figsize=(7.6, 3.5), layout='constrained')
    for i, (tau,color) in enumerate([(.012,BLUE),(.2,ORANGE)]):
        part = c[c.tau == tau].set_index('route').loc[['go','trigger','stop']]
        y = part.delta_stop_pp.to_numpy()
        ax.errorbar(np.arange(3)+(i-.5)*.16, y, yerr=np.stack([y-part.stop_MC95_low_pp,part.stop_MC95_high_pp-y]), fmt='o', color=color, capsize=4,label=f'{1000*tau:g} ms')
    ax.axhline(0,color='gray',lw=.7)
    ax.set(xticks=range(3),xticklabels=['Go','Cue trigger','Post-trigger stop'],ylabel='Stopping contrast (percentage points)',xlabel='Modulated process')
    ax.legend(title='Chemical lifetime')
    save(fig,'fig3')
    # Confirm the six main contrasts directly from stored event times.
    trial_checks=[]
    for _,row in c.iterrows():
        values=[]
        for seed in [26091711,26091712,26091713]:
            a=np.load(HERE/'data/synaptic-veto/trials'/f'{row.condition_id}_{seed}.npz')
            times=a['times'][2,2]  # temporal, top allocation
            stop=np.isfinite(times[:,:,2]) & (times[:,:,2]<times[:,:,0])
            values.extend((stop[1].astype(int)-stop[0].astype(int)).tolist())
        val=100*math.fsum(values)/len(values)
        assert len(values)==24576 and abs(val-row.delta_stop_pp)<1e-10
        trial_checks.append({'condition':row.condition_id,'n':len(values),'delta_stop_pp':val})

    selected=csv('tables/population_selected.csv'); gain=csv('tables/circuit_gain.csv')
    fig,ax=plt.subplots(1,2,figsize=(8.4,3.7),layout='constrained')
    labels={'baseline':'Reference','b':'HQ wait\n0.1/s','cp':'P recovery\n0.1/s','ce':'E oxidation\n10000/s','closed_shell_noise_scale':'Noise\n×100'}
    ax[0].bar(range(len(selected)),100*selected.delta_yield,color=BLUE)
    ax[0].set(xticks=range(len(selected)),xticklabels=[labels[a] for a in selected.axis],ylabel='Yield contrast (percentage points)',title='(a) Molecular sensitivity')
    ax[0].tick_params(axis='x',labelsize=8);ax[0].axhline(0,color='gray',lw=.7)
    y=gain.delta_stop_pp.to_numpy()
    ax[1].errorbar(gain.gain,y,yerr=np.stack([y-gain.stop_MC95_low_pp,gain.stop_MC95_high_pp-y]),fmt='o-',color=ORANGE,capsize=3,label='3072 pairs')
    conf=c[(c.route=='stop')&(c.tau==.2)].iloc[0]
    ax[1].errorbar([1024],[conf.delta_stop_pp],yerr=[[conf.delta_stop_pp-conf.stop_MC95_low_pp],[conf.stop_MC95_high_pp-conf.delta_stop_pp]],fmt='s',color=BLUE,capsize=3,label='24576 pairs')
    ax[1].set(xlabel='Assumed gain',ylabel='Stopping contrast (percentage points)',title='(b) Gain sensitivity');ax[1].legend(fontsize=8)
    save(fig,'fig4')

    hk=csv('tables/hk_heldout_uncertainty.csv')
    pred=csv('tables/hk_prediction_rows.csv')
    scales=json.loads((HERE/'data/hk-redox-update/fits.json').read_text())['scales']
    rmse={}; diffs={}
    for split in ['fit','cell_validation','condition_validation']:
        errors={}
        for model in ['dynamic','static']:
            part=pred[(pred.split==split)&(pred.model==model)].copy()
            part['err']=((part.predicted_log_ratio-part.observed_log_ratio)/part['mode'].map(scales))**2
            per=part.groupby(['cell_id','mode']).err.mean().groupby('cell_id').mean()
            errors[model]=per;rmse[split,model]=float(np.sqrt(per.mean()))
        diffs[split]=float((errors['dynamic']-errors['static']).mean())
    for _,row in hk.iterrows():
        assert abs(diffs[row['split']]-row.mean_dynamic_minus_static_MSE)<1e-10
    fig,ax=plt.subplots(1,2,figsize=(8.4,3.6),layout='constrained',gridspec_kw={'width_ratios':[1,1.15]})
    for i,model in enumerate(['dynamic','static']):
        ax[0].bar(np.arange(3)+(i-.5)*.34,[rmse[s,model] for s in ['fit','cell_validation','condition_validation']],width=.34,label=model.capitalize(),color=[BLUE,ORANGE][i])
    ax[0].set(xticks=range(3),xticklabels=['Training','Held-out','40 min'],ylabel='Standardised RMSE',title='(a) Predictive error');ax[0].legend(fontsize=8)
    y=hk.mean_dynamic_minus_static_MSE.to_numpy()
    ax[1].errorbar([0,1],y,yerr=np.stack([y-hk.interval_low,hk.interval_high-y]),fmt='o',color=BLUE,capsize=4)
    ax[1].axhline(0,color='gray',ls='--');ax[1].set(xticks=[0,1],xticklabels=['Held-out\n16 cells','40 min\n7 cells'],ylabel='Dynamic − static mean squared error',title='(b) Cell-bootstrap uncertainty',xlim=(-.5,1.5))
    save(fig,'fig5')

    bound_rows=[]
    for N in [1,6,26,27,100,1000,10000]:
        p=binom.pmf(np.arange(N+1),N,q0); q=binom.pmf(np.arange(N+1),N,q1)
        tv=.5*np.abs(p-q).sum();tv2=.5*np.abs(recurrence(N,q0)-recurrence(N,q1)).sum()
        assert abs(tv-tv2)<1e-11
        bound_rows.append({'N':N,'TV_pp':100*tv,'paired_bound_pp':-100*np.expm1(N*np.log1p(-dq))})
    report={'input_hashes_checked':len(hashes),'scope':'Frozen-operator reconstruction, saved-trial reclassification and replotting; not an upstream simulation rerun',
            'yields':yields,'delta_q':dq,'half_decay_s':half,'correlation_share_percent':float(100*w[[2,5,6,7]].sum()/w.sum()),
            'chemical_residuals':analytic.tolist(),'trial_checks':trial_checks,
            'hk_rmse':{f'{s}/{m}':v for (s,m),v in rmse.items()},'hk_mse_differences':diffs,'bounds':bound_rows,'status':'passed'}
    (OUT/'checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'passed','figures':5,'input_hashes':len(hashes),'delta_q':dq,'half_decay_s':half}))


if __name__=='__main__':
    main()
