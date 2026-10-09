# Rotational constraints on cryptochrome nuclear population memory

Computational source companion to **Rotational constraints on nuclear population memory in a cryptochrome reaction model** (Journal of Biological Physics draft, 9 October 2026, `jbp-motion-2.0.0`).

**Hikaru Wakaura**  
QIRI (Quantum Integrated Research Institute Inc.), Tokyo 107-0061, Japan  
h.wakaura@qiri.co.jp

[Manuscript PDF](paper/manuscript.pdf) · [Supplement PDF](paper/supplement.pdf)

The current revision compares free rotation, protection of chemical waits, reversible binding, bound angular fluctuations, electric orientation, electronic EFG/hyperfine response and deformation of an assumed rotational binding well. The earlier fixed-orientation storage result is a conditional control. The neural calculation illustrates readout under prescribed gain and timing; it does not establish a native brain pathway or effect size. Electric-field-updated yields have not been propagated to new neural trials.

Nuclear-population inheritance has prior theoretical precedents, including [Wong et al. (2021)](https://doi.org/10.1063/5.0038947). `population` retains joint diagonal probabilities and correlations at the specified HQ boundary while removing boundary coherence. Both it and `full` retain within-cycle quantum dynamics. Population is not a complex wavefunction amplitude.

## Repository contents

This repository contains calculation source, numerical tests, the two compiled paper PDFs and minimal project metadata. It deliberately excludes simulation outputs, numerical result tables, trial records, frozen operators, MD trajectories, electronic checkpoints, downloaded papers, plots, LaTeX sources and submission/data archives. The manuscript PDFs contain the figures and results reported in the article.

**Online Resource 2 is a separate numerical supplement and is not in this repository.** There is no public data deposit or data DOI. The allowlist in `.gitignore` prevents generated data from being added by ordinary staging. Earlier source modules are retained for provenance; the two PDFs and the motion-revision reproduction script are the current article artifacts.

## Setup and checks without paper data

Use Python 3.11 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Default tests cover the earlier chemical-routing, Hk, pulse and stochastic-count routines, plus the new bound-wobble and electric-orientation generators. They check independent solutions, limiting cases, conservation, noise factors, equilibrium moments and the covariant rotational generator without reading excluded paper data.

Three additional field-coupling checks use synthetic inputs only:

```bash
python -m pytest -q \
  tests/test_hq_field_couplings.py::test_well_no_field_and_detailed_balance_limit \
  tests/test_hq_field_couplings.py::test_well_curvature_matches_independent_finite_difference \
  tests/test_hq_field_couplings.py::test_batched_lindblad_matches_independent_kron
```

Other field-coupling and population tests require the omitted frozen operator archive. `python -m pytest tests` explicitly selects those as well. Passing the data-free tests does not reproduce the paper or validate the assumed biological parameters.

The base dependency versions describe the source-export validation environment. Original neural trial generation uses **NumPy 2.5.0**, enforced by its drivers; do not rerun it with another NumPy version and claim identical random streams. Bounded reproduction instead reads the saved discordance counts and verifies the intervals without rerunning neural trials. Electronic response additionally requires **PySCF 2.13.0** and **pyscf-properties 0.1.0**, together with the separate checkpoints, basis and embedding inputs. These are optional dependencies, not part of the data-free test environment.

## Current calculation map

| Calculation | Implementation | Drivers |
|---|---|---|
| Common rigid-body rotation and reaction cycle | `impl/hq_rotation.py` | `investigations/2026-10-09-hq-rotation/` |
| Protected waits, rapid renewal and added storage | `impl/hq_rotation_cases.py` | `investigations/2026-10-09-rotation-cases/` |
| Reversible capture/release and fast bound wobble | `impl/hq_binding.py` | `investigations/2026-10-09-binding-motion/` |
| Electric orientational bias and axial limits | `impl/hq_electric.py` | `investigations/2026-10-09-membrane-field/` |
| Electronic EFG/HFC susceptibility and well deformation | `impl/hq_field_couplings.py`, `impl/dark_electronic_response.py` | `investigations/2026-10-09-field-couplings/` |
| Conditional stochastic neural readout | `impl/synaptic_veto.py`, `impl/sparse_synaptic.py`, `impl/stochastic_branch_spike.py` | `neural.py` in the binding-motion and rotation-cases directories |
| Current article figure/table reproduction | `paper/journal-of-biological-physics/motion-revision/reproduce.py` | Same file, with the separate numerical supplement |

The original drivers retain their workspace-relative paths. Shell wrappers describe the original orchestration and may default to the original local Python path or runtime directory; inspect their interpreter settings before use. Drivers that create reports or validate original provenance may also require excluded reports/manifests. No data are automatically downloaded or fabricated.

A standalone conditional well can be evaluated without paper inputs:

```bash
python - <<'PYTHON'
from impl.hq_field_couplings import well
w = well([0., 0., 1.], 5., 1e-11, 10.910941626120245)
print({'rms_angle_deg': w['rms_deg'], 'correlation_times_s': w['taus'].tolist()})
PYTHON
```

This computes the specified mathematical restraint; its parameters are not fitted native binding quantities.

## Inputs and reproduction limits

**This source-only repository cannot reproduce every reported number by itself.** The small tests above run without paper data; original numerical runs require separately supplied inputs.

| Calculation | Required external inputs |
|---|---|
| Rotation, binding and field spin cycles | `data/population-memory/latest_case.npz` containing frozen H/R/E Hamiltonians and local bath operators |
| Initial operator construction | State-specific electronic results, `data/dark-md-coefficients/` trajectory/metadata, atom/basis and embedding specifications; see `investigations/2026-09-16-dark-basis-resolution/propagate.py` |
| Matched H/R electronic response | `data/dark-embedded-tensors/` anionic HQ and oxidised radius-12 Å checkpoints, geometries and MM environments |
| Matched E electronic response | `data/dark-basis-resolution/upcj2N__full/` neutral-SQ checkpoint and full minimum-image environment, including the nitrogen basis |
| Conditional neural reproduction | Appropriate molecular result JSON files plus the archived NumPy 2.5.0 environment; raw event verification additionally needs saved trial arrays |
| Current figures, tables and intervals | Extracted Online Resource 2, with its `data/`, `audit/input-manifest.json` and `tables/` beside `reproduce.py` |

The separate compact Online Resource 2 supports arithmetic checks and figure/table regeneration from frozen results. It does not contain all upstream MD trajectories, electronic checkpoints or neural trial arrays and is not an end-to-end first-principles reproduction bundle. It is intentionally absent from this public repository.

The current figure script generates PDF/SVG figures, PNG audit previews, LaTeX table fragments and an arithmetic/interval check report. These are outputs, remain ignored, and should not be committed. Numerical tables display five digits after the decimal point; stored inputs and calculations retain full precision. Display precision is not a statement of physical accuracy.

## Earlier source retained for provenance

Earlier independent modules include `impl/population_memory.py`, `impl/chemical_carrier_mapping.py`, `impl/literature_bridge_recalculation.py` and `impl/hk_redox_memory.py`, with their September investigation drivers. They are not newly calibrated serial stages of the current neural example. `impl/reservoir.py` and `impl/spin.py` retain earlier reference calculations. The prior figure script remains under `paper/physical-biology/supplementary-data/reproduce.py`; it is not the current four-figure manuscript entry point.

The old current-data reanalysis requires the original Rorsman et al. source workbook `41586_2025_8734_MOESM4_ESM.xlsx` from [the primary article](https://doi.org/10.1038/s41586-025-08734-4), plus provenance inputs referenced by its driver. The reader is included; the workbook is not redistributed.

## License

Source code (`impl/`, `investigations/`, `tests/`, `evidence/**/*.py`, `paper/**/*.py`) is released under the [MIT License](LICENSE). The manuscript and supplement PDFs (`paper/manuscript.pdf`, `paper/supplement.pdf`) are licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Third-party data are not redistributed and remain under their original terms.
