#!/usr/bin/env bash
# tmux supervises Slurm allocations; Python only executes inside Slurm.
set -euo pipefail
DIFFUSCENE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$DIFFUSCENE_ROOT"
DIFFUSCENE_RUN="${DIFFUSCENE_RUN:-networks/diffuscene/runs/livingroom_uncond_100k}"
export DIFFUSCENE_CONFIG="${DIFFUSCENE_CONFIG:-training_configs/diffuscene_livingroom_uncond_100k.yaml}"
DIFFUSCENE_JOB_NAME="${DIFFUSCENE_JOB_NAME:-diffuscene-living-uncond}"
mkdir -p "$DIFFUSCENE_RUN" networks/diffuscene/slurm
exec 8>"$DIFFUSCENE_RUN/pipeline.lock"
flock -n 8 || { echo 'DiffuScene pipeline already running'; exit 1; }
exec > >(tee -a "$DIFFUSCENE_RUN/pipeline.log") 2>&1
[[ -f datasets/atiss_livingroom_obb_v2_wo_lat/READY ]] || { echo 'Preprocessing must complete first'; exit 1; }
[[ -f "$DIFFUSCENE_RUN/VERIFIED" ]] || { echo 'GPU integration verification must complete first'; exit 1; }
if [[ -f "$DIFFUSCENE_RUN/COMPLETE" ]]; then
  echo 'This DiffuScene run is already complete.'
  exit 0
fi
while true; do
  echo "[$(date -Is)] Submitting a separate one-GPU allocation (48 hours, resume last.pt)."
  if sbatch --parsable --wait --job-name="$DIFFUSCENE_JOB_NAME" networks/diffuscene/train_livingroom.slurm | tee "$DIFFUSCENE_RUN/train_job.txt"; then
    [[ -f "$DIFFUSCENE_RUN/COMPLETE" ]] || { echo 'Training exited without a completion marker'; exit 1; }
    echo "[$(date -Is)] Training completed."
    break
  else
    DIFFUSCENE_EXIT=$?
  fi
  DIFFUSCENE_JOB="$(cut -d ';' -f 1 "$DIFFUSCENE_RUN/train_job.txt")"
  DIFFUSCENE_STATE="$(sacct -j "$DIFFUSCENE_JOB" -X -n -o State -P | head -n 1)"
  if [[ -f "$DIFFUSCENE_RUN/last.pt" ]] && \
     { [[ "$DIFFUSCENE_EXIT" -eq 75 ]] || [[ "$DIFFUSCENE_STATE" == TIMEOUT* || "$DIFFUSCENE_STATE" == PREEMPTED* ]]; }; then
    echo "[$(date -Is)] Resuming after $DIFFUSCENE_JOB ($DIFFUSCENE_STATE, exit $DIFFUSCENE_EXIT)."
  else
    echo "Stopped after $DIFFUSCENE_JOB ($DIFFUSCENE_STATE, exit $DIFFUSCENE_EXIT); inspect the Slurm log."
    exit "$DIFFUSCENE_EXIT"
  fi
done
