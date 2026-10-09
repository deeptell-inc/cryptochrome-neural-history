#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
field_python=${ROTATION_PYTHON:-/opt/anaconda3/bin/python}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$field_python" -m pytest tests/test_hq_electric.py tests/test_hq_binding.py -q
"$field_python" investigations/2026-10-09-membrane-field/run.py --bound
"$field_python" investigations/2026-10-09-membrane-field/run.py --L 3 --kappas 10.910941626120245 --bound
"$field_python" investigations/2026-10-09-membrane-field/spectrum.py
"$field_python" investigations/2026-10-09-membrane-field/axial_limit.py
"$field_python" investigations/2026-10-09-membrane-field/verify.py
"$field_python" investigations/2026-10-09-membrane-field/finalize.py
"$field_python" investigations/2026-10-09-membrane-field/report.py
"$field_python" investigations/2026-10-09-membrane-field/audit.py
