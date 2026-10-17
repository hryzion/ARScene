#!/usr/bin/env bash
set -euo pipefail
DIFFUSCENE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$DIFFUSCENE_ROOT"
DIFFUSCENE_PYTHON="${DIFFUSCENE_PYTHON:-/home/cscgbnu/.conda/envs/arscene/bin/python}"
exec "$DIFFUSCENE_PYTHON" -m pip install --disable-pip-version-check --no-deps \
  --target "$DIFFUSCENE_ROOT/.runtime/diffuscene" einops==0.8.1
