#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
rotation_python=${ROTATION_PYTHON:-/opt/anaconda3/bin/python}
neural_python=${NEURAL_PYTHON:-/Users/deeptell01/Documents/alterego/personal/myvenv/bin/python}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# Numerical sources and the archived frozen tensor NPZ are required.
# Existing per-case result files are reused; use a fresh copied workspace to
# force regeneration without deleting any prior scientific outputs.
for angle in 0 1 5; do
 "$rotation_python" investigations/2026-10-09-binding-motion/run.py --angle "$angle"
done
"$rotation_python" investigations/2026-10-09-binding-motion/run.py --angle 0.1 --verify
"$rotation_python" investigations/2026-10-09-binding-motion/batch.py --cutoff 2 --angles 1 --tau-w 1e-10
"$rotation_python" investigations/2026-10-09-binding-motion/batch.py --cutoff 3 --angles 0 1 5
"$rotation_python" investigations/2026-10-09-binding-motion/quadrature_check.py
"$rotation_python" investigations/2026-10-09-binding-motion/verify.py
"$rotation_python" investigations/2026-10-09-binding-motion/augment.py --angle 0
"$rotation_python" investigations/2026-10-09-binding-motion/augment.py --angle 1
"$rotation_python" -m pytest tests/test_hq_binding.py -q
"$neural_python" investigations/2026-10-09-binding-motion/neural.py
"$rotation_python" investigations/2026-10-09-binding-motion/finalize.py
