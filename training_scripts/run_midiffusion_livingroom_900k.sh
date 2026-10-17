#!/usr/bin/env bash
# Keep the 900k-step job running across 48-hour Slurm allocations.
set -euo pipefail
MID_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MID_ROOT"
MID_RUN=pretrained/midiffusion/livingroom_unconditional_900k
mkdir -p "$MID_RUN/slurm"
exec 8>"$MID_RUN/pipeline.lock"
flock -n 8 || { echo 'A MiDiffusion pipeline is already running'; exit 1; }
exec > >(tee -a "$MID_RUN/pipeline.log") 2>&1
[[ -f datasets/atiss_livingroom_obb_v2_wo_lat/READY ]] || { echo 'Dataset is not ready'; exit 1; }
if [[ -f "$MID_RUN/COMPLETE" ]]; then
  echo 'MiDiffusion livingroom 900k has already completed.'
  exit 0
fi
while true; do
  echo "[$(date -Is)] Submitting one GPU; automatically restore last.pt when present."
  if sbatch --parsable --wait training_scripts/midiffusion_livingroom_900k.sbatch | tee "$MID_RUN/train_job.txt"; then
    [[ -f "$MID_RUN/COMPLETE" ]] || { echo 'Training exited without completing the budget'; exit 1; }
    echo "[$(date -Is)] MiDiffusion completed 900000 steps."
    break
  else
    MID_EXIT=$?
  fi
  MID_JOB="$(cut -d ';' -f 1 "$MID_RUN/train_job.txt")"
  MID_STATE="$(sacct -j "$MID_JOB" -X -n -o State -P | head -n 1)"
  if [[ -f "$MID_RUN/last.pt" ]] && \
     { [[ "$MID_EXIT" -eq 75 ]] || [[ "$MID_STATE" == TIMEOUT* || "$MID_STATE" == PREEMPTED* ]]; }; then
    echo "[$(date -Is)] Resuming after job=$MID_JOB state=$MID_STATE exit=$MID_EXIT."
  else
    echo "MiDiffusion stopped: job=$MID_JOB state=$MID_STATE exit=$MID_EXIT. Inspect the Slurm log."
    exit "$MID_EXIT"
  fi
done
