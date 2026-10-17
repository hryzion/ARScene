#!/usr/bin/env bash
set -euo pipefail
NORMALIZER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$NORMALIZER_ROOT"
NORMALIZER_PYTHON="${RELTREE_PYTHON:-/home/cscgbnu/.conda/envs/arscene/bin/python}"
export NORMALIZER_TEST_DEVICE=cuda:0
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/reltree-matplotlib}"
exec "$NORMALIZER_PYTHON" -u -m unittest discover -s tests -p test_scene_token_normalizer.py -v
