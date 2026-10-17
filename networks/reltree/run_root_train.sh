#!/usr/bin/env bash
set -euo pipefail
RELTREE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$RELTREE_ROOT"
export RELTREE_CONFIG="${RELTREE_CONFIG:-training_configs/reltree_root_ar.yaml}"
export RELTREE_RUN_DIR="${RELTREE_RUN_DIR:-networks/reltree/runs/bedroom_root_ar_12x768_attr256_cd10}"
exec bash networks/reltree/run_full_train.sh
