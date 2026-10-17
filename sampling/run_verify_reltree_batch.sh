#!/usr/bin/env bash
set -euo pipefail
SAMPLING_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SAMPLING_ROOT"
SAMPLING_PYTHON="${SAMPLING_PYTHON:-/home/cscgbnu/.conda/envs/arscene/bin/python}"
export PYTHONPATH="$SAMPLING_ROOT/.runtime/diffuscene${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/arscene-sampling-matplotlib}"
export OPENBLAS_NUM_THREADS=1
export RELTREE_TEST_DEVICE=cuda:0
"$SAMPLING_PYTHON" -u -m unittest networks.reltree.tests.test_batch_generation networks.reltree.tests.test_room_root -v
exec "$SAMPLING_PYTHON" -u -m sampling.verify_reltree_batch "$@"
