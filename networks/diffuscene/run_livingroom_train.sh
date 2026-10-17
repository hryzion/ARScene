#!/usr/bin/env bash
set -euo pipefail
DIFFUSCENE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$DIFFUSCENE_ROOT"
DIFFUSCENE_PYTHON="${DIFFUSCENE_PYTHON:-/home/cscgbnu/.conda/envs/arscene/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/diffuscene-matplotlib}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$DIFFUSCENE_ROOT/.runtime/diffuscene${PYTHONPATH:+:$PYTHONPATH}"
exec "$DIFFUSCENE_PYTHON" -u train_diffuscene.py \
  --config "${DIFFUSCENE_CONFIG:-training_configs/diffuscene_livingroom_uncond_100k.yaml}" --cuda 0
