#!/usr/bin/env bash
set -euo pipefail
RELTREE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$RELTREE_ROOT"
RELTREE_RUN_DIR="${RELTREE_RUN_DIR:-networks/reltree/runs/bedroom_ar_8x512_cd10}"
RELTREE_DATA_ROOT="${RELTREE_DATA_ROOT:-datasets/processed_reltree_bedroom_v2}"
RELTREE_CONFIG="${RELTREE_CONFIG:-training_configs/reltree_full.yaml}"
[[ -f "$RELTREE_CONFIG" ]] || { echo "Training config not found: $RELTREE_CONFIG"; exit 1; }
[[ -f "$RELTREE_DATA_ROOT/READY" ]] || { echo "Corrected dataset is not ready"; exit 1; }
mkdir -p "$RELTREE_RUN_DIR"
# Prevent a second submission from writing this run's checkpoints concurrently.
exec 9>"$RELTREE_RUN_DIR/training.lock"
flock -n 9 || { echo "Another training process owns $RELTREE_RUN_DIR"; exit 1; }
RELTREE_RESUME_ARGS=()
if [[ -f "$RELTREE_RUN_DIR/last.pt" ]]; then
  echo "Resuming from $RELTREE_RUN_DIR/last.pt"
  RELTREE_RESUME_ARGS=(--resume "$RELTREE_RUN_DIR/last.pt")
fi
exec bash networks/reltree/run_train.sh \
  --config "$RELTREE_CONFIG" \
  --data-root "$RELTREE_DATA_ROOT" --output "$RELTREE_RUN_DIR" "${RELTREE_RESUME_ARGS[@]}"
