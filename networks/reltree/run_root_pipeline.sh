#!/usr/bin/env bash
# Run in tmux; Slurm performs all GPU work and the pipeline resumes timeouts.
set -euo pipefail
RELTREE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$RELTREE_ROOT"
export RELTREE_CONFIG="training_configs/reltree_root_ar.yaml"
export RELTREE_RUN_DIR="networks/reltree/runs/bedroom_root_ar_12x768_attr256_cd10"
export RELTREE_TRAIN_SLURM="networks/reltree/train_root_reltree.slurm"
export RELTREE_VERIFIED_JOB_ID="${RELTREE_VERIFIED_JOB_ID:-16329}"
export RELTREE_AFTER_JOB=""
exec bash networks/reltree/run_full_pipeline.sh
