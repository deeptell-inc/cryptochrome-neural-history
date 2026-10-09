#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
coupling_python=${FIELD_PYTHON:-/opt/anaconda3/bin/python}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD/.runtime/field-electronic${PYTHONPATH:+:$PYTHONPATH}"
# Local runtime: PySCF 2.13.0, pyscf-properties 0.1.0; see requirements.txt.
# Needs the saved electronic checkpoints, MM environment, MD trajectory and
# frozen population-memory archive. This is not a data-free public export.
for state in H R E; do
  "$coupling_python" investigations/2026-10-09-field-couplings/electronic.py "$state"
  "$coupling_python" investigations/2026-10-09-field-couplings/electronic.py "$state" --finite
done
"$coupling_python" investigations/2026-10-09-field-couplings/compare_finite.py
"$coupling_python" -m pytest tests/test_hq_field_couplings.py tests/test_hq_electric.py tests/test_hq_binding.py -q > data/field-couplings/tests.log
"$coupling_python" investigations/2026-10-09-field-couplings/wells.py
"$coupling_python" investigations/2026-10-09-field-couplings/verify_wells.py
for mode in baseline well electronic combined; do
  "$coupling_python" investigations/2026-10-09-field-couplings/run.py --mode "$mode"
done
"$coupling_python" investigations/2026-10-09-field-couplings/run.py --mode combined --L 3
"$coupling_python" investigations/2026-10-09-field-couplings/verify_cycle.py
"$coupling_python" investigations/2026-10-09-field-couplings/spectrum.py
"$coupling_python" investigations/2026-10-09-field-couplings/finalize.py
"$coupling_python" investigations/2026-10-09-field-couplings/report.py
"$coupling_python" investigations/2026-10-09-field-couplings/audit.py
