#!/usr/bin/env bash
set -euo pipefail
DIFFUSCENE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$DIFFUSCENE_ROOT"
DIFFUSCENE_PYTHON="${DIFFUSCENE_PYTHON:-/home/cscgbnu/.conda/envs/arscene/bin/python}"
export PYTHONPATH="$DIFFUSCENE_ROOT/.runtime/diffuscene${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/diffuscene-matplotlib}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$DIFFUSCENE_PYTHON" -u -m networks.diffuscene.audit_training_alignment
