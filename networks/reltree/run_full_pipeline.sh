#!/usr/bin/env bash
# Run this inside tmux. Actual computation is always submitted through Slurm.
set -euo pipefail
RELTREE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$RELTREE_ROOT"
export RELTREE_RUN_DIR="${RELTREE_RUN_DIR:-networks/reltree/runs/bedroom_ar_8x512_cd10}"
export RELTREE_DATA_ROOT="${RELTREE_DATA_ROOT:-datasets/processed_reltree_bedroom_v2}"
export RELTREE_CONFIG="${RELTREE_CONFIG:-training_configs/reltree_full.yaml}"
RELTREE_TRAIN_SLURM="${RELTREE_TRAIN_SLURM:-networks/reltree/train_full_reltree.slurm}"
RELTREE_AFTER_JOB="${RELTREE_AFTER_JOB:-}"
[[ -f "$RELTREE_CONFIG" && -f "$RELTREE_TRAIN_SLURM" ]] || { echo "Missing config or training Slurm script"; exit 1; }
if [[ -n "$RELTREE_AFTER_JOB" && ! "$RELTREE_AFTER_JOB" =~ ^[0-9]+$ ]]; then
  echo "RELTREE_AFTER_JOB must be a numeric Slurm job ID"; exit 1
fi
RELTREE_PROCESSED_ROOT="${RELTREE_PROCESSED_ROOT:-datasets/processed_bedroom_wo_lat_v2}"
mkdir -p networks/reltree/slurm "$RELTREE_RUN_DIR"
exec 8>"$RELTREE_RUN_DIR/pipeline.lock"
flock -n 8 || { echo "A pipeline is already running for $RELTREE_RUN_DIR"; exit 1; }
exec > >(tee -a "$RELTREE_RUN_DIR/pipeline.log") 2>&1

submit_and_wait() {
  local job_record="$1"
  shift
  sbatch --parsable --wait "$@" | tee "$job_record"
}

echo "[$(date -Is)] Full bedroom pipeline started."
if [[ -n "${RELTREE_VERIFIED_JOB_ID:-}" ]]; then
  [[ "$(sacct -j "$RELTREE_VERIFIED_JOB_ID" -X -n -o State -P | head -n 1)" == COMPLETED ]] || {
    echo "Specified verification job did not complete successfully"; exit 1;
  }
  printf '%s\n' "$RELTREE_VERIFIED_JOB_ID" > "$RELTREE_RUN_DIR/verify_job.txt"
  echo "Using successful verification job $RELTREE_VERIFIED_JOB_ID."
else
  submit_and_wait "$RELTREE_RUN_DIR/verify_job.txt" networks/reltree/verify_resume.slurm
fi
if [[ ! -f "$RELTREE_PROCESSED_ROOT/READY" ]]; then
  echo "Regenerating upstream room_data.npz with preprocess.py from Levels2021."
  submit_and_wait "$RELTREE_RUN_DIR/upstream_job.txt" preprocess_script/preprocess_levels.slurm \
    --raw-root datasets/Levels2021 --filter_fn bedroom \
    --split-root datasets/processed_bedroom_wo_lat --asset-center-tolerance .003 --output "$RELTREE_PROCESSED_ROOT"
fi
if [[ -f "$RELTREE_DATA_ROOT/READY" ]]; then
  echo "Reusing completed conversion: $RELTREE_DATA_ROOT"
else
  echo "Submitting full train/validation conversion."
  submit_and_wait "$RELTREE_RUN_DIR/prepare_job.txt" networks/reltree/prepare_reltree.slurm \
    --processed-root "$RELTREE_PROCESSED_ROOT" --output "$RELTREE_DATA_ROOT"
fi
echo "[$(date -Is)] Upstream preprocessing and RelTree conversion finished."

RELTREE_DEPENDENCY_ARGS=()
if [[ -n "$RELTREE_AFTER_JOB" ]]; then
  RELTREE_DEPENDENCY_ARGS=(--dependency="afterok:$RELTREE_AFTER_JOB" --kill-on-invalid-dep=yes)
  printf '%s\n' "$RELTREE_AFTER_JOB" > "$RELTREE_RUN_DIR/after_job.txt"
  echo "Waiting for successful completion of job $RELTREE_AFTER_JOB before training."
fi
while true; do
  echo "[$(date -Is)] Submitting training (48 hours, automatic last.pt recovery)."
  if submit_and_wait "$RELTREE_RUN_DIR/train_job.txt" "${RELTREE_DEPENDENCY_ARGS[@]}" "$RELTREE_TRAIN_SLURM"; then
    echo "[$(date -Is)] Training completed."
    break
  else
    RELTREE_JOB_EXIT=$?
  fi
  # The first allocation already satisfied the predecessor dependency. Resume
  # jobs should not depend on a job ID that Slurm may eventually purge.
  RELTREE_DEPENDENCY_ARGS=()
  RELTREE_JOB_ID="$(cut -d ';' -f 1 "$RELTREE_RUN_DIR/train_job.txt")"
  RELTREE_JOB_STATE="$(sacct -j "$RELTREE_JOB_ID" -X -n -o State -P | head -n 1)"
  if [[ -f "$RELTREE_RUN_DIR/last.pt" ]] && \
     { [[ "$RELTREE_JOB_EXIT" -eq 75 ]] || [[ "$RELTREE_JOB_STATE" == TIMEOUT* || "$RELTREE_JOB_STATE" == PREEMPTED* ]]; }; then
    echo "[$(date -Is)] Job $RELTREE_JOB_ID stopped with exit=$RELTREE_JOB_EXIT state=$RELTREE_JOB_STATE; submitting its saved checkpoint again."
  else
    echo "Job $RELTREE_JOB_ID stopped: exit=$RELTREE_JOB_EXIT state=$RELTREE_JOB_STATE. Inspect its Slurm log before restarting."
    exit "$RELTREE_JOB_EXIT"
  fi
done
