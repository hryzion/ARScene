#!/usr/bin/env bash
# Supervise Slurm allocations in tmux; all model computation stays inside Slurm.
set -euo pipefail
ATISS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ATISS_ROOT"
ATISS_RUN=pretrained/atiss/diningroom_roomshape_12x768_room64_500k
ATISS_PYTHON="${ATISS_PYTHON:-/home/cscgbnu/.conda/envs/diffuscene/bin/python}"
mkdir -p "$ATISS_RUN/slurm"
exec 8>"$ATISS_RUN/pipeline.lock"
flock -n 8 || { echo 'An ATISS pipeline is already running'; exit 1; }
exec > >(tee -a "$ATISS_RUN/pipeline.log") 2>&1
[[ -f datasets/atiss_diningroom_obb_v2_wo_lat/READY ]] || { echo 'ATISS dataset is not ready'; exit 1; }
[[ -f datasets/atiss_diningroom_obb_v2_wo_lat/VALIDATED ]] || { echo 'ATISS dataset is not validated'; exit 1; }
if [[ -f "$ATISS_RUN/COMPLETE" ]]; then
  echo 'ATISS diningroom roomshape 12x768 500k has already completed.'
  exit 0
fi
while true; do
  echo "[$(date -Is)] Submitting one GPU for up to 48 hours; resume last.pt when present."
  if sbatch --parsable --wait training_scripts/atiss_diningroom_roomshape_12x768_500k.sbatch | tee "$ATISS_RUN/train_job.txt"; then
    "$ATISS_PYTHON" - "$ATISS_RUN" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
last = json.loads((root / 'metrics.jsonl').read_text().strip().splitlines()[-1])
if last.get('event') != 'complete' or last.get('step') != 500000:
    raise RuntimeError('Training exited without reaching the 500000-step budget')
(root / 'COMPLETE').write_text(json.dumps(last) + '\n')
PY
    echo "[$(date -Is)] ATISS roomshape completed 500000 steps."
    break
  else
    ATISS_EXIT=$?
  fi
  ATISS_JOB="$(cut -d ';' -f 1 "$ATISS_RUN/train_job.txt")"
  ATISS_STATE="$(sacct -j "$ATISS_JOB" -X -n -o State -P | head -n 1)"
  if [[ -f "$ATISS_RUN/last.pt" ]] && \
     { [[ "$ATISS_EXIT" -eq 75 ]] || [[ "$ATISS_STATE" == TIMEOUT* || "$ATISS_STATE" == PREEMPTED* ]]; }; then
    echo "[$(date -Is)] Resuming after job=$ATISS_JOB state=$ATISS_STATE exit=$ATISS_EXIT."
  else
    echo "ATISS stopped: job=$ATISS_JOB state=$ATISS_STATE exit=$ATISS_EXIT. Inspect the Slurm log."
    exit "$ATISS_EXIT"
  fi
done
