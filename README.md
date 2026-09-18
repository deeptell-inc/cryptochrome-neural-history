# Cryptochrome spin-state history and neural responses

Computational source companion to **Conditions for transmitting cryptochrome spin-state history to neural responses** (Physical Biology submission draft).

**Hikaru Wakaura**  
QIRI (Quantum Integrated Research Institute Inc.), Tokyo 107-0061, Japan  
h.wakaura@qiri.co.jp

[Manuscript PDF](paper/manuscript.pdf) · [Supplement PDF](paper/supplement.pdf)

Repository: [https://github.com/deeptell-inc/cryptochrome-neural-history](https://github.com/deeptell-inc/cryptochrome-neural-history). The numerical Supplementary Data archive is a separate submission artifact and is not included in this repository.

## Scope

This repository contains the compiled manuscript and supplement, the principal calculation modules and their local Python dependencies, simulation/analysis drivers, and numerical tests. It is a focused paper companion, not an archive of every earlier investigation. `impl/reservoir.py` and `impl/spin.py` retain the earlier observable-history reference calculations.

Precomputed results, trial records, frozen operators, MD trajectories, electronic-structure checkpoints, downloaded articles, workbooks, plots, LaTeX sources and submission archives are excluded. The PDFs contain the figures and results presented in the paper. `.gitignore` uses an allowlist to prevent generated results from being added accidentally.

The five modules are separate conditional tests, not successive stages of a calibrated biological pathway. The circuit module uses the spin-model yield contrast directly; it does not derive that contrast through the chemical-routing and Hk-storage modules. `population_HQ` retains density-matrix populations (squared amplitudes) at the HQ cycle boundary while removing boundary coherence; it still shares within-reaction quantum dynamics with `full`.

## Setup and data-free checks

Use Python 3.11 or later. From this repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The default pytest configuration selects the chemical-routing, Hk-memory, three-state pulse and stochastic-count tests that do not read excluded data. These compare implementations with independent ODE solutions, matrix exponentials, limiting cases, finite differences and stochastic moments. They do not reproduce the entire paper. Dependency versions in `requirements.txt` describe the export-validation environment, not the original simulation environment.

Additional data-free circuit checks:

```bash
python -m pytest -q \
  tests/test_sparse_synaptic.py::test_allocation_equal_budget_and_selective_block \
  tests/test_sparse_synaptic.py::test_exact_transition_against_independent_ode \
  tests/test_synaptic_veto.py::test_finite_pool_against_independent_master_equation \
  tests/test_synaptic_veto.py::test_shared_uniform_correlation_and_exact_majority
```

Running `python -m pytest tests` explicitly selects all bundled tests, including tests that require the omitted input/result files.

The source modules also support calculations with user-specified parameters, without paper data. For example, this computes a hypothetical three-state Hk pulse and prints its peak; its inputs are scenario choices, not measured neural parameters:

```bash
python - <<'PYTHON'
from impl.literature_bridge_recalculation import solve_pulse, peak
solution = solve_pulse(a0=10., off=1., binding=100., tau=.05)
time_s, occupancy = peak(solution)
print({'peak_time_s': time_s, 'peak_oxidized_fraction': occupancy})
PYTHON
```

## Model and driver map

| Component | Numerical source | Original driver |
|---|---|---|
| M1: spin history and population storage | `impl/population_memory.py`, `impl/dark_*` | `investigations/2026-09-16-population-memory/run.py` |
| M2: chemical product routing | `impl/chemical_carrier_mapping.py` | `investigations/2026-09-17-chemical-carrier-mapping/run.py` |
| M3: three-state Hk storage | `impl/literature_bridge_recalculation.py` | `investigations/2026-09-18-literature-bridge-recalculation/run.py` |
| M4: conditional synaptic circuit | `impl/synaptic_veto.py`, `impl/sparse_synaptic.py`, `impl/stochastic_branch_spike.py` | `investigations/2026-09-17-synaptic-veto/run.py` |
| M5: published current-data reanalysis | `impl/hk_redox_memory.py` | `investigations/2026-09-17-hk-redox-update/run.py` |
| Supplemental bounds and tables | `paper/2026-09-18/revision_analysis.py` | same file |
| Article figures and frozen-data checks | `paper/physical-biology/supplementary-data/reproduce.py` | same file |

## External inputs and reproduction limits

**This source-only repository cannot reproduce all published numerical values by itself.** The original drivers are retained for inspection and reuse. They expect the original workspace-relative paths, and some enforce hash checks on archived inputs, audit manifests and earlier reports. Those files are intentionally absent; restoring only a CSV is not sufficient for every driver. Low-level functions can be used with independently supplied inputs. There is no fabricated replacement data or automatic download step.

| Calculation | Inputs that must be supplied separately |
|---|---|
| M1 operator construction | `data/dark-basis-resolution/*/result.json`, embedded/electronic tensors, `data/dark-md-coefficients/` trajectories and metadata, topology and electronic input specifications |
| M1 frozen-operator tests | `data/population-memory/latest_case.npz`, `modes_and_states.npz`, associated baseline and verification tables |
| M2/M3/M4 original scans | `data/dark-basis-resolution/conditional_products_spikes.csv`; fitted/scenario records and inherited preserved-input/audit manifests referenced by the drivers |
| M5 reanalysis | Original Rorsman et al. source workbook `41586_2025_8734_MOESM4_ESM.xlsx` under `evidence/2026-09-17-native-noise-literature/`, plus archived files required by the driver's provenance checks |
| Supplemental bounds/tables | Corresponding `data/` outputs and figures referenced by `revision_analysis.py` |
| Article figure reproduction | `data/`, `tables/` and `source-hashes.json` from the separate supplementary-data bundle, placed beside `reproduce.py` |

The M5 workbook is identified by the primary article DOI [10.1038/s41586-025-08734-4](https://doi.org/10.1038/s41586-025-08734-4). It is not redistributed here. Its reader is included as source code.

Electronic-structure functions additionally require **PySCF and its `pyscf.prop` extensions**. The pcJ-2 nitrogen basis file, molecular coordinates/checkpoints and trajectory inputs are also external. These optional calculations are not covered by the data-free checks or base dependency file. Upstream molecular-dynamics generation is outside this focused export.

After restoring the required inputs at their expected paths, run drivers from the repository root, for example:

```bash
mkdir -p data
python investigations/2026-09-17-hk-redox-update/run.py
python investigations/2026-09-17-hk-redox-update/verify_and_plot.py
```

Some drivers perform large scans and create results under `data/`; the figure reproduction script writes to its adjacent `reproduced/` directory. Review their parameters and input requirements before running them.

Numerical sources are copied from the research workspace without changes to model equations or parameter values. The only source portability adjustment replaces the machine-specific plotting-cache directory in `verify_and_plot.py` with the operating system temporary directory.

## License

Source code (`impl/`, `investigations/`, `tests/`, `evidence/**/*.py`, `paper/**/*.py`) is released under the [MIT License](LICENSE). The manuscript and supplement PDFs (`paper/manuscript.pdf`, `paper/supplement.pdf`) are licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Third-party source data referenced above are not redistributed and remain under their original terms.
