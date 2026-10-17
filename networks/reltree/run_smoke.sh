#!/usr/bin/env bash
set -euo pipefail
RELTREE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$RELTREE_ROOT"
RELTREE_PYTHON="${RELTREE_PYTHON:-/home/cscgbnu/.conda/envs/arscene/bin/python}"
RELTREE_SMOKE_DATA="${RELTREE_SMOKE_DATA:-datasets/processed_reltree_bedroom_smoke_v2}"
RELTREE_RUN_DIR="${RELTREE_RUN_DIR:-networks/reltree/runs/smoke-${SLURM_JOB_ID:-local}}"
export RELTREE_TEST_DEVICE=cuda:0

"$RELTREE_PYTHON" -u -m networks.reltree.gpu_probe
"$RELTREE_PYTHON" -m unittest discover -s networks/reltree/tests -v
if [[ ! -f "$RELTREE_SMOKE_DATA/tokenizer.json" ]]; then
  "$RELTREE_PYTHON" -u -m networks.reltree.prepare --output "$RELTREE_SMOKE_DATA" --train-limit 32 --val-limit 8
fi
"$RELTREE_PYTHON" -u train_reltree.py --config training_configs/reltree_smoke.yaml \
  --data-root "$RELTREE_SMOKE_DATA" --output "$RELTREE_RUN_DIR/train" --steps 120
"$RELTREE_PYTHON" -u train_reltree.py --config training_configs/reltree_smoke.yaml \
  --data-root "$RELTREE_SMOKE_DATA" --output "$RELTREE_RUN_DIR/overfit" --steps 200 --overfit-single
"$RELTREE_PYTHON" -u train_reltree.py --config training_configs/reltree_smoke.yaml \
  --data-root "$RELTREE_SMOKE_DATA" --output "$RELTREE_RUN_DIR/resume" --steps 205 --overfit-single \
  --resume "$RELTREE_RUN_DIR/overfit/last.pt"
