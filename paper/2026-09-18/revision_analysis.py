"""PRX Life revision: frozen-data tables and conditional distinguishability bounds.

No native biological calibration, new fit, or modification of simulation inputs.
Run with Python containing numpy, scipy, pandas and matplotlib.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import shutil

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cry-paper-mpl")
import numpy as np
import pandas as pd
from scipy.stats import binom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "revision-v1.1.0"
TABLES = OUT / "tables"
TABLES.mkdir(parents=True, exist_ok=True)
FIG = HERE / "figures"
INPUTS = {}


def read_csv(rel):
    p = ROOT / rel
    INPUTS[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return pd.read_csv(p)


def read_json(rel):
    p = ROOT / rel
    INPUTS[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())


def table(df, name):
    df.to_csv(TABLES / name, index=False)


def markdown(df):
    # No optional tabulate dependency.
    def fmt(x):
        if isinstance(x, (float, np.floating)):
            return f"{x:.6g}"
        return str(x).replace("|", "\\|")
    return "\n".join([
        "| " + " | ".join(df.columns) + " |",
        "|" + "---|" * len(df.columns),
        *["| " + " | ".join(fmt(x) for x in row) + " |"
          for row in df.itertuples(index=False, name=None)],
    ])


def binomial_recurrence(N, p):
    """Independent normalized recurrence starting at the mode, not scipy.pmf."""
    m = min(N, math.floor((N + 1) * p))
    a = np.zeros(N + 1)
    a[m] = 1.
    for k in range(m, N):
        a[k + 1] = a[k] * (N-k)/(k+1) * p/(1-p)
    for k in range(m, 0, -1):
        a[k - 1] = a[k] * k/(N-k+1) * (1-p)/p
    return a/a.sum()


base = read_csv("data/population-memory/baseline_comparison.csv").set_index("model")
# Use the actual D4 source values (the D1 reset differs only by solver residual).
src = read_csv("data/dark-basis-resolution/conditional_products_spikes.csv")
src = src[(src.condition == "upcj2N__full") & (src.gamma_oxygen_s == 0)].set_index("model")
q0, q1 = float(src.loc["full", "yield_reference"]), float(src.loc["full", "yield_P"])
dq = q1-q0
assert 0 < q0 < q1 < 1

factors = read_csv("data/population-memory/factor_scans.csv")
table(factors, "population_factor_scans_all_216.csv")
full = factors[factors.model == "full"]
assert len(full) == 72
selected = full[((full.axis == "baseline") |
                 ((full.axis == "cp") & (full.value == .1)) |
                 ((full.axis == "ce") & (full.value == 10000)) |
                 ((full.axis == "b") & (full.value == .1)) |
                 ((full.axis == "closed_shell_noise_scale") & (full.value == 100)))][
                     ["axis", "value", "delta_yield"]].copy()
table(selected, "population_selected.csv")
modes = read_csv("data/population-memory/write_read_modes.csv")
modes = modes[(modes.stage == "after_20_cycles") & (modes.gamma_oxygen_s == 0)].copy()
assert len(modes) == 8
modes["yield_share_percent"] = 100*modes.contribution_at_zero/modes.contribution_at_zero.sum()
table(modes, "population_modes_8.csv")
assert abs(modes.contribution_at_zero.sum() - (base.loc["full", "yield_P"]-q0)) < 1e-8

circuit = read_csv("data/synaptic-veto/summary.csv")
table(circuit, "circuit_summary_all_1932.csv")
table(circuit[circuit.group == "grid"], "circuit_grid_432.csv")
table(read_csv("data/synaptic-veto/supplement.csv"), "circuit_extra_controls_540.csv")
gain = circuit[circuit.condition_id.isin(["179_grid", "180_grid", "181_grid", "182_grid"])
               & (circuit.integration == "temporal") & (circuit.allocation == "top")].sort_values("gain")
assert len(gain) == 4
gain = gain[["condition_id", "gain", "n", "delta_stop_pp", "stop_MC95_low_pp", "stop_MC95_high_pp"]]
table(gain, "circuit_gain.csv")
confirmation = circuit[(circuit.group == "confirmation") & (circuit.integration == "temporal")
                       & (circuit.allocation == "top")]
assert len(confirmation) == 6
table(confirmation, "circuit_confirmations_6.csv")

verify = read_json("data/hk-redox-update/verification.json")
hk = pd.DataFrame([{k: r[k] for k in ["split", "cells", "mean_dynamic_minus_static_MSE"]}
                   | {"interval_low": r["cluster_percentile_interval"][0],
                      "interval_high": r["cluster_percentile_interval"][1]}
                   for r in verify["heldout_model_comparison"]])
table(hk, "hk_heldout_uncertainty.csv")
boot = read_csv("data/hk-redox-update/cell_bootstrap.csv")
expanded = read_csv("data/hk-redox-update/expanded_bound_bootstrap.csv")
bs = read_json("data/hk-redox-update/bootstrap_summary.json")
params = list(bs["percentile_intervals"])
quant = []
for par in params:
    qs = np.quantile(boot[par], [.025, .5, .975])
    qe = np.quantile(expanded[par], [.025, .5, .975])
    assert np.max(np.abs(qs - bs["percentile_intervals"][par])) < 1e-9
    quant.append(dict(parameter=par, q025=qs[0], median=qs[1], q975=qs[2],
                      expanded_q975=qe[2], near_upper_bound=bs["near_upper_bounds"][par]))
table(pd.DataFrame(quant), "hk_parameter_intervals.csv")
for rel in ["prediction_rows.csv", "observations.csv", "occupation_family.json", "data_provenance.json"]:
    path = ROOT/"data/hk-redox-update"/rel
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    shutil.copy2(path, TABLES/("hk_"+rel))

for rel in ["phase_only_control.csv", "nonsecular_counterexample.csv", "generator_coupling.csv",
            "projected_wait_audit.csv", "reaction_cycle_coupling.csv"]:
    path = ROOT/"data/population-memory"/rel
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    shutil.copy2(path, TABLES/("population_"+rel))
for folder, name, dest in [("data/hk-redox-update", "current_envelopes.csv", TABLES/"hk_current_envelopes.csv"),
                           ("data/population-memory", "population_memory.png", FIG/"population_memory.png"),
                           ("data/population-memory", "population_memory.svg", FIG/"population_memory.svg")]:
    path = ROOT/folder/name
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    shutil.copy2(path, dest)

# Two different bounds. Marginal TV applies only to a common readout kernel
# whose entire intervention-dependent input is this count. The D4 simulator
# gates the source with a reference path before arrival, so its joint history
# is not certified by marginal TV alone. The paired-event bound is safe for
# the frozen stochastic D4 code: identical counts imply identical future arms.
bounds = []
errors = []
for N in [1, 6, 10, 26, 27, 100, 1000, 10000]:
    n = np.arange(N+1)
    p0, p1 = binom.pmf(n, N, q0), binom.pmf(n, N, q1)
    r0, r1 = binomial_recurrence(N, q0), binomial_recurrence(N, q1)
    tv = .5*np.abs(p1-p0).sum()
    tv2 = .5*np.abs(r1-r0).sum()
    errors.append(abs(tv-tv2))
    paired_bound = -np.expm1(N*np.log1p(-dq))
    # Condition on the baseline count in the implemented nested binomial draw.
    change_given_n = -np.expm1((N-n)*np.log1p(-dq/(1-q0)))
    paired2 = float(np.sum(r0 * change_given_n))
    errors.append(abs(paired_bound-paired2))
    assert tv <= paired_bound+1e-12
    bounds.append(dict(N=N, single_count_common_kernel_TV_pp=100*tv,
                       D4_paired_event_bound_pp=100*paired_bound))
bd = pd.DataFrame(bounds)
table(bd, "conditional_bounds.csv")
assert np.all(np.isfinite(errors)) and max(errors) < 1e-11

# Direct enumeration of all two-arm events for a small, independent test.
Ntest=4
joint=np.zeros((Ntest+1,Ntest+1))
for a in range(Ntest+1):
    for b in range(a,Ntest+1):
        joint[a,b]=binom.pmf(a,Ntest,q0)*binom.pmf(b-a,Ntest-a,dq/(1-q0))
assert np.max(abs(joint.sum(0)-binom.pmf(np.arange(Ntest+1),Ntest,q1))) < 1e-12
assert abs(1-np.trace(joint)-(-np.expm1(Ntest*np.log1p(-dq)))) < 1e-12

# Counterexample to applying count-only TV in the presence of side information:
# the second arm's observation (baseline_count, arm_count) can distinguish
# diagonal from off-diagonal pairs. This intentionally exceeds marginal TV.
sideinfo_example = {"N":Ntest,"joint_event_difference":float(1-np.trace(joint)),
                    "marginal_count_TV":float(.5*np.abs(joint.sum(0)-joint.sum(1)).sum())}
assert sideinfo_example["joint_event_difference"] > sideinfo_example["marginal_count_TV"]

plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
                     "axes.spines.top":False,"axes.spines.right":False})
blue, orange = "#0072B2", "#D55E00"
fig, ax = plt.subplots(1, 2, figsize=(11.8, 4.6), layout="constrained")
ax[0].bar(np.arange(len(selected)), selected.delta_yield*100, color=blue)
labels={"baseline":"Baseline", "b":"HQ wait\n0.1 / s", "cp":"P recovery\n0.1 / s",
        "ce":"E oxidation\n10,000 / s", "closed_shell_noise_scale":"Closed-shell\nnoise x100"}
ax[0].set(xticks=np.arange(len(selected)), xticklabels=[labels[a] for a in selected.axis],
          ylabel="Yield contrast (percentage points)",title="A. Stored one-factor counterexamples")
ax[0].tick_params(axis="x",labelsize=9)
ax[0].axhline(0,color="gray",lw=.8)
g=gain.delta_stop_pp.to_numpy()
ax[1].errorbar(gain.gain,g,yerr=np.stack([g-gain.stop_MC95_low_pp,gain.stop_MC95_high_pp-g]),
               fmt="o-",color=orange,capsize=3,label="Grid: 3,072 paired trials")
conf=confirmation[(confirmation.route=="stop") & (confirmation.tau==.2)].iloc[0]
ax[1].errorbar([conf.gain],[conf.delta_stop_pp],yerr=[[conf.delta_stop_pp-conf.stop_MC95_low_pp],
               [conf.stop_MC95_high_pp-conf.delta_stop_pp]],fmt="s",color=blue,capsize=3,label="Confirmation: 24,576 pairs")
ax[1].set(xlabel="Assumed gain",ylabel="Stopping contrast (percentage points)",title="B. Stop route, top 8, 200 ms")
ax[1].legend(fontsize=8)
fig.suptitle("Parameter dependence; scan settings are not biological priors")
for ext in ["png","svg","pdf"]:
    fig.savefig(FIG/f"robustness_revision.{ext}",dpi=180,bbox_inches="tight")
plt.close(fig)

fig, ax=plt.subplots(figsize=(7.5,4.3),layout="constrained")
x=hk.mean_dynamic_minus_static_MSE.to_numpy()
ax.errorbar(x,[1,0],xerr=np.stack([x-hk.interval_low,hk.interval_high-x]),fmt="o",color=blue,capsize=4)
ax.axvline(0,color="gray",ls="--")
ax.set(yticks=[1,0],yticklabels=["Held-out cells (16)","Held-out 40 min (7)"],
       xlabel="Mean cell-balanced MSE difference: dynamic minus static",
       title="Hk predictive contrast remains unresolved",ylim=(-.5,1.5))
fig.text(.5,-.06,"Stored cell-cluster percentile intervals; conditional on model, baselines and independent cells.\nAnimal / batch dependence is not quantified.",ha="center",fontsize=9)
for ext in ["png","svg","pdf"]:
    fig.savefig(FIG/f"hk_uncertainty_revision.{ext}",dpi=180,bbox_inches="tight")
plt.close(fig)

sections=[("全８モード（20周期後、gamma_oxygen=0）",modes[["mode","decay_rate_s","time_constant_s","write_amplitude","read_sensitivity","yield_share_percent"]]),
          ("本文に採録する符号・振幅感度",selected),("低利得の保存格子（確認試行と区別）",gain),
          ("Hk留保誤差差と保存区間",hk),("Hkパラメータの保存bootstrap",pd.DataFrame(quant)),
          ("異なる条件に対する二つの上限",bd)]
report="# v1.1.0 補足数表\n\n保存データの再集計と、新しい条件付き数学的上限。生体較正ではない。\n\n"
report+="\n\n".join("## "+name+"\n\n"+markdown(df) for name,df in sections)
report+="\n\n全216因子行、全432格子行、全1932集計行、追加対照540行、Hk観測・予測のCSVは同じtablesディレクトリに同梱した。行数は独立標本数ではない。TV列は独立な単時点共通読出しを仮定し、履歴を読むD4回路全体への上限として使わない。D4には別列の対応試行上限を用いる。詳細な導出と適用条件は補足S13を参照。\n"
(OUT/"supplement_tables.md").write_text(report)
audit={"scope":"Frozen-data extraction and conditional mathematical bounds, not native calibration",
       "q0":q0,"q1":q1,"delta_q":dq,"max_independent_bound_error":max(errors),
       "side_information_counterexample":sideinfo_example,"bounds":bounds,
       "minimum_binomial_slots_for_14_16015625pp_by_D4_bound":math.ceil(np.log1p(-.1416015625)/np.log1p(-dq)),
       "factor_settings":len(full),"factor_rows":len(factors),"grid_rows":int((circuit.group=="grid").sum()),
       "input_sha256":INPUTS,"files":[str(p.relative_to(HERE)) for p in TABLES.glob('*')]}
(OUT/"analysis-checks.json").write_text(json.dumps(audit,indent=2)+"\n")
print(json.dumps({"factor_settings":len(full),"grid_rows":audit["grid_rows"],"bound_error":max(errors),
                  "bound_minimum_N":audit["minimum_binomial_slots_for_14_16015625pp_by_D4_bound"]},indent=2))
