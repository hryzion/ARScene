#!/usr/bin/env bash
# Supervise separate one-GPU Slurm allocations; automatically resume last.pt.
set -euo pipefail
DIFFUSCENE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIFFUSCENE_ROOT"
DIFFUSCENE_RUN=networks/diffuscene/runs/diningroom_uncond_500k_aug
mkdir -p "$DIFFUSCENE_RUN/slurm"
exec 8>"$DIFFUSCENE_RUN/pipeline.lock"
flock -n 8 || { echo 'A DiffuScene pipeline is already running'; exit 1; }
exec > >(tee -a "$DIFFUSCENE_RUN/pipeline.log") 2>&1
[[ -f datasets/atiss_diningroom_obb_v2_wo_lat/READY && -f datasets/atiss_diningroom_obb_v2_wo_lat/VALIDATED ]] || {
  echo 'Diningroom preprocessing and validation must complete first'; exit 1;
}
if [[ -f "$DIFFUSCENE_RUN/COMPLETE" ]]; then
  echo 'DiffuScene diningroom 500k has already completed.'
  exit 0
fi
while true; do
  echo "[$(date -Is)] Submitting one GPU for up to 48 hours; automatically restore last.pt when present."
  if sbatch --parsable --wait training_scripts/diffuscene_diningroom_uncond_500k.sbatch | tee "$DIFFUSCENE_RUN/train_job.txt"; then
    [[ -f "$DIFFUSCENE_RUN/COMPLETE" ]] || { echo 'Training exited without completing the budget'; exit 1; }
    echo "[$(date -Is)] DiffuScene completed 500000 steps."
    break
  else
    DIFFUSCENE_EXIT=$?
  fi
  DIFFUSCENE_JOB="$(cut -d ';' -f 1 "$DIFFUSCENE_RUN/train_job.txt")"
  DIFFUSCENE_STATE="$(sacct -j "$DIFFUSCENE_JOB" -X -n -o State -P | head -n 1)"
  if [[ -f "$DIFFUSCENE_RUN/last.pt" ]] && \
     { [[ "$DIFFUSCENE_EXIT" -eq 75 ]] || [[ "$DIFFUSCENE_STATE" == TIMEOUT* || "$DIFFUSCENE_STATE" == PREEMPTED* ]]; }; then
    echo "[$(date -Is)] Resuming after job=$DIFFUSCENE_JOB state=$DIFFUSCENE_STATE exit=$DIFFUSCENE_EXIT."
  else
    echo "DiffuScene stopped: job=$DIFFUSCENE_JOB state=$DIFFUSCENE_STATE exit=$DIFFUSCENE_EXIT. Inspect the Slurm log."
    exit "$DIFFUSCENE_EXIT"
  fi
done
