#!/usr/bin/env bash
#SBATCH --job-name=atiss-sample-collision
#SBATCH --partition=general
#SBATCH --qos=advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/cscgbnu/zhx/code/ARScene
#SBATCH --output=/home/cscgbnu/zhx/code/ARScene/sampling/logs/%x-%j.out
#SBATCH --error=/home/cscgbnu/zhx/code/ARScene/sampling/logs/%x-%j.out

set -euo pipefail
ATISS_SAMPLE_RUN="${1:?Pass the sampling run directory containing checkpoint.pt and manifest.json}"
ATISS_SAMPLE_TAG="${2:-best}"
[[ "$ATISS_SAMPLE_TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Tag must contain only letters, digits, underscores or hyphens'; exit 1; }
ATISS_SAMPLE_PYTHON=/home/cscgbnu/.conda/envs/diffuscene/bin/python
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/arscene-atiss-sampling-matplotlib
export PYTHONUNBUFFERED=1
exec 9>"$ATISS_SAMPLE_RUN/sample.lock"
flock -n 9 || { echo 'Sampling is already running in this directory'; exit 1; }
[[ ! -d "$ATISS_SAMPLE_RUN/layouts" ]] || { echo 'Use a fresh sampling directory'; exit 1; }
echo "[$(date -Is)] job=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"$ATISS_SAMPLE_PYTHON" -u test_atiss.py \
  --checkpoint "$ATISS_SAMPLE_RUN/checkpoint.pt" --output "$ATISS_SAMPLE_RUN" \
  --tag "$ATISS_SAMPLE_TAG" --num 1000 --seed 42 --device cuda:0 --layout-only
"$ATISS_SAMPLE_PYTHON" sampling/atiss_layouts_to_scenes.py \
  --src "$ATISS_SAMPLE_RUN/layouts/$ATISS_SAMPLE_TAG" --output "$ATISS_SAMPLE_RUN/scenes" --expected-count 1000
"$ATISS_SAMPLE_PYTHON" compute_collision.py --src "$ATISS_SAMPLE_RUN/scenes" --collision-only \
  --output "$ATISS_SAMPLE_RUN/collision_metrics.json" --details "$ATISS_SAMPLE_RUN/collision_details.jsonl"
date -Is > "$ATISS_SAMPLE_RUN/COMPLETE"
