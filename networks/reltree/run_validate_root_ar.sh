#!/usr/bin/env bash
set -euo pipefail
RELTREE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$RELTREE_ROOT"
RELTREE_PYTHON="${RELTREE_PYTHON:-/home/cscgbnu/.conda/envs/arscene/bin/python}"
RELTREE_RUN_DIR="${RELTREE_RUN_DIR:-networks/reltree/runs/validation-root-ar-${SLURM_JOB_ID:-local}}"
exec "$RELTREE_PYTHON" -u -m networks.reltree.validate_root_ar --output "$RELTREE_RUN_DIR" "$@"
