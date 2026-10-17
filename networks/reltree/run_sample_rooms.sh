#!/usr/bin/env bash
set -euo pipefail
RELTREE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$RELTREE_ROOT"
RELTREE_PYTHON="${RELTREE_PYTHON:-/home/cscgbnu/.conda/envs/arscene/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/reltree-matplotlib}"
export RELTREE_TEST_DEVICE=cuda:0
"$RELTREE_PYTHON" -u -m networks.reltree.gpu_probe
"$RELTREE_PYTHON" -u -m unittest networks.reltree.tests.test_condition_dropout networks.reltree.tests.test_model.GenerationTests -v
exec "$RELTREE_PYTHON" -u -m networks.reltree.sample_rooms "$@"
