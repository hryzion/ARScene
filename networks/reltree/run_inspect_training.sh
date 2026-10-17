#!/usr/bin/env bash
set -euo pipefail
RELTREE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$RELTREE_ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/reltree-matplotlib}"
RELTREE_PYTHON="${RELTREE_PYTHON:-/home/cscgbnu/.conda/envs/arscene/bin/python}"
exec "$RELTREE_PYTHON" -u -m networks.reltree.inspect_training "$@"
