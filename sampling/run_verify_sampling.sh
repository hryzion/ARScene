#!/usr/bin/env bash
set -euo pipefail
SAMPLING_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SAMPLING_ROOT"
SAMPLING_PYTHON="${SAMPLING_PYTHON:-/home/cscgbnu/.conda/envs/arscene/bin/python}"
export PYTHONPATH="$SAMPLING_ROOT/.runtime/diffuscene${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/arscene-sampling-matplotlib}"
export OPENBLAS_NUM_THREADS=1
exec "$SAMPLING_PYTHON" -u -m sampling.verify_sampling
