#!/usr/bin/env bash
set -euo pipefail
RELTREE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$RELTREE_ROOT"
RELTREE_PYTHON="${RELTREE_PYTHON:-/home/cscgbnu/.conda/envs/arscene/bin/python}"
RELTREE_LIVING_SOURCE="${RELTREE_LIVING_SOURCE:-datasets/atiss_livingroom_obb_v2_wo_lat}"
RELTREE_LIVING_OUTPUT="${RELTREE_LIVING_OUTPUT:-datasets/processed_reltree_livingroom_v2}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
bash networks/reltree/run_prepare.sh \
  --processed-root "$RELTREE_LIVING_SOURCE" \
  --output "$RELTREE_LIVING_OUTPUT" --include-test
"$RELTREE_PYTHON" -u -m networks.reltree.audit_prepared \
  --source "$RELTREE_LIVING_SOURCE" --output "$RELTREE_LIVING_OUTPUT"
