#!/usr/bin/env bash
set -euo pipefail
RELTREE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$RELTREE_ROOT"
export RELTREE_CONFIG="training_configs/reltree_diningroom_root_ar_200k.yaml"
export RELTREE_RUN_DIR="networks/reltree/runs/diningroom_root_ar_12x768_attr256_cd50_step_200k"
export RELTREE_DATA_ROOT="datasets/processed_reltree_diningroom_v2"
export OPENBLAS_NUM_THREADS=1
[[ -f "$RELTREE_DATA_ROOT/VALIDATED" ]] || { echo 'Diningroom data must pass validation first'; exit 1; }
exec bash networks/reltree/run_full_train.sh
