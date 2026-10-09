#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
rotation_python=${ROTATION_PYTHON:-/opt/anaconda3/bin/python}
neural_python=${NEURAL_PYTHON:-/Users/deeptell01/Documents/alterego/personal/myvenv/bin/python}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# Requires the already validated inputs/results from 2026-10-09-hq-rotation.
"$rotation_python" investigations/2026-10-09-rotation-cases/molecular.py --cutoff 2
"$rotation_python" investigations/2026-10-09-rotation-cases/molecular.py --cutoff 3 --validation-only
"$rotation_python" investigations/2026-10-09-rotation-cases/verify_molecular.py
"$rotation_python" investigations/2026-10-09-rotation-cases/protected_storage.py
"$rotation_python" investigations/2026-10-09-rotation-cases/chemistry.py
for task_batch in mixture routes fast field; do
 "$neural_python" investigations/2026-10-09-rotation-cases/neural.py --batch "$task_batch" --n 8192
done
"$neural_python" investigations/2026-10-09-rotation-cases/summarize.py
"$neural_python" investigations/2026-10-09-rotation-cases/plot.py
"$neural_python" investigations/2026-10-09-rotation-cases/write_report.py
"$neural_python" investigations/2026-10-09-rotation-cases/finalize.py
