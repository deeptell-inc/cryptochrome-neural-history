#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
task_python=${PYTHON:-/opt/anaconda3/bin/python}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$task_python" investigations/2026-10-09-hq-rotation/validate.py
for task_tau in 9.72 21.476595751657293 99.12; do
  for task_cutoff in 2 3; do
    "$task_python" investigations/2026-10-09-hq-rotation/run.py --cutoff "$task_cutoff" --tau-ns "$task_tau" --check-aligned
  done
  "$task_python" investigations/2026-10-09-hq-rotation/independent_readout.py --cutoff 4 --tau-ns "$task_tau"
done
"$task_python" investigations/2026-10-09-hq-rotation/independent_readout.py --cutoff 3 --tau-ns 21.476595751657293
for task_cutoff in 5 6; do
  "$task_python" investigations/2026-10-09-hq-rotation/independent_readout.py --cutoff "$task_cutoff" --tau-ns 99.12
done
"$task_python" investigations/2026-10-09-hq-rotation/run.py --cutoff 1
"$task_python" investigations/2026-10-09-hq-rotation/run.py --cutoff 0 --field-uT 0
"$task_python" investigations/2026-10-09-hq-rotation/spectrum.py
"$task_python" investigations/2026-10-09-hq-rotation/summarize.py
