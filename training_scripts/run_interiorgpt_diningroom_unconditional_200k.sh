#!/usr/bin/env bash
# Supervise Slurm allocations; resume model, optimizer, RNG and data cursor.
set -euo pipefail
INTERIORGPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$INTERIORGPT_ROOT"
INTERIORGPT_OUTPUT=pretrained/interiorGPT/diningroom_unconditional_multicode_12x768_200k
mkdir -p "$INTERIORGPT_OUTPUT/slurm"
exec 8>"$INTERIORGPT_OUTPUT/pipeline.lock"
flock -n 8 || { echo 'An InteriorGPT pipeline is already running'; exit 1; }
exec > >(tee -a "$INTERIORGPT_OUTPUT/pipeline.log") 2>&1
[[ -f datasets/processed_diningroom_wo_lat/normalizer_stats.json ]] || {
  echo 'The diningroom normalizer required by the pretrained VQ-VAE is missing'; exit 1;
}
[[ -f pretrained/vqvae/residual_vq_diningroom_wo_lat_vocab2048/vqvae.pth ]] || {
  echo 'The diningroom 2048-entry VQ-VAE checkpoint is missing'; exit 1;
}
if [[ -f "$INTERIORGPT_OUTPUT/COMPLETE" ]]; then
  [[ "$(cat "$INTERIORGPT_OUTPUT/COMPLETE")" == 200000 ]] || { echo 'Unexpected completed budget'; exit 1; }
  echo 'InteriorGPT diningroom 200k has already completed.'
  exit 0
fi
while true; do
  echo "[$(date -Is)] Submitting one Slurm GPU for up to 48 hours; automatically resume last.pt."
  if sbatch --parsable --wait training_scripts/interiorgpt_diningroom_unconditional_200k.sbatch | tee "$INTERIORGPT_OUTPUT/train_job.txt"; then
    [[ -f "$INTERIORGPT_OUTPUT/COMPLETE" && "$(cat "$INTERIORGPT_OUTPUT/COMPLETE")" == 200000 ]] || {
      echo 'Training exited without completing 200000 steps'; exit 1;
    }
    echo "[$(date -Is)] InteriorGPT completed 200000 steps."
    break
  else
    INTERIORGPT_EXIT=$?
  fi
  INTERIORGPT_JOB="$(cut -d ';' -f 1 "$INTERIORGPT_OUTPUT/train_job.txt")"
  INTERIORGPT_STATE="$(sacct -j "$INTERIORGPT_JOB" -X -n -o State -P | head -n 1)"
  if [[ -f "$INTERIORGPT_OUTPUT/last.pt" ]] && \
     { [[ "$INTERIORGPT_EXIT" -eq 75 ]] || [[ "$INTERIORGPT_STATE" == TIMEOUT* || "$INTERIORGPT_STATE" == PREEMPTED* ]]; }; then
    echo "[$(date -Is)] Resuming after job=$INTERIORGPT_JOB state=$INTERIORGPT_STATE exit=$INTERIORGPT_EXIT."
  else
    echo "Stopped: job=$INTERIORGPT_JOB state=$INTERIORGPT_STATE exit=$INTERIORGPT_EXIT; inspect the Slurm log."
    exit "$INTERIORGPT_EXIT"
  fi
done
