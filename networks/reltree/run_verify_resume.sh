#!/usr/bin/env bash
set -euo pipefail
RELTREE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$RELTREE_ROOT"
RELTREE_PYTHON="${RELTREE_PYTHON:-/home/cscgbnu/.conda/envs/arscene/bin/python}"
export RELTREE_TEST_DEVICE=cuda:0
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/reltree-matplotlib}"
"$RELTREE_PYTHON" -u -m networks.reltree.gpu_probe
"$RELTREE_PYTHON" -u -m unittest discover -s networks/reltree/tests -v
