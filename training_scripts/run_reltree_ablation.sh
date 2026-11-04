#!/usr/bin/env bash
# One supervisor per output; resume only after a time limit, preemption or safe signal.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RELTREE_CONFIG="${1:?Usage: bash training_scripts/run_reltree_ablation.sh CONFIG}"
RELTREE_PYTHON="${RELTREE_PYTHON:-/mnt/atlas/cv/hxzhu/.venvs/arscene-preprocess-py311/bin/python}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
RELTREE_OUTPUT="$("$RELTREE_PYTHON" -B -c 'import sys; from pathlib import Path; from networks.reltree.ablations.config import load_config; print(Path(load_config(sys.argv[1])["training"]["output"]).resolve())' "$RELTREE_CONFIG")"
mkdir -p "$RELTREE_OUTPUT/slurm"
exec 8>"$RELTREE_OUTPUT/pipeline.lock"
flock -n 8 || { echo 'This output already has a supervisor'; exit 1; }
exec > >(tee -a "$RELTREE_OUTPUT/pipeline.log") 2>&1
while true; do
  if sbatch --parsable --wait --job-name=reltree-ablation \
      --output="$RELTREE_OUTPUT/slurm/%x-%j.out" --error="$RELTREE_OUTPUT/slurm/%x-%j.out" \
      training_scripts/reltree_ablation.sbatch "$RELTREE_CONFIG" | tee "$RELTREE_OUTPUT/job.txt"; then
    "$RELTREE_PYTHON" -B - "$RELTREE_CONFIG" "$RELTREE_OUTPUT" <<'PY'
import json,sys
from pathlib import Path
from networks.reltree.ablations.config import load_config
config = load_config(sys.argv[1])
report = json.loads((Path(sys.argv[2]) / 'completed.json').read_text())
assert report['status'] == 'complete' and report['step'] == config['training']['max_steps'], report
assert report['variant'] == config['ablation']['variant'], report
PY
    break
  else
    RELTREE_EXIT=$?
  fi
  RELTREE_JOB="$(cut -d ';' -f 1 "$RELTREE_OUTPUT/job.txt")"
  RELTREE_STATE="$(sacct -j "$RELTREE_JOB" -X -n -o State -P | head -n 1 || true)"
  if [[ -f "$RELTREE_OUTPUT/last.pt" ]] && \
      { [[ "$RELTREE_EXIT" -eq 75 ]] || [[ "$RELTREE_STATE" == TIMEOUT* || "$RELTREE_STATE" == PREEMPTED* ]]; }; then
    echo "Resuming job=$RELTREE_JOB state=$RELTREE_STATE exit=$RELTREE_EXIT"
  else
    echo "Stopped job=$RELTREE_JOB state=$RELTREE_STATE exit=$RELTREE_EXIT"
    exit "$RELTREE_EXIT"
  fi
done
