#!/usr/bin/env bash
set -euo pipefail
DIFFUSCENE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$DIFFUSCENE_ROOT"
DIFFUSCENE_PYTHON="${DIFFUSCENE_PYTHON:-/home/cscgbnu/.conda/envs/arscene/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/diffuscene-matplotlib}"
export OPENBLAS_NUM_THREADS=1
exec "$DIFFUSCENE_PYTHON" networks/diffuscene/plot_loss.py "$@"
