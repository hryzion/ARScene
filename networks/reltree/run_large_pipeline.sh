#!/usr/bin/env bash
# Run in tmux; sbatch enforces ordering, and the pipeline resumes interruptions.
set -euo pipefail
RELTREE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$RELTREE_ROOT"
if (( $# > 1 )); then
  echo "Usage: bash networks/reltree/run_large_pipeline.sh [predecessor_job_id]"; exit 2
fi
RELTREE_PREVIOUS_RUN="networks/reltree/runs/bedroom_ar_8x512_cd10"
export RELTREE_CONFIG="training_configs/reltree_12x768.yaml"
export RELTREE_RUN_DIR="networks/reltree/runs/bedroom_ar_12x768_attr256_cd10"
export RELTREE_TRAIN_SLURM="networks/reltree/train_large_reltree.slurm"
export RELTREE_AFTER_JOB="${1:-$(cut -d ';' -f 1 "$RELTREE_PREVIOUS_RUN/train_job.txt")}"
export RELTREE_VERIFIED_JOB_ID="$(cat "$RELTREE_PREVIOUS_RUN/verify_job.txt")"
exec bash networks/reltree/run_full_pipeline.sh
