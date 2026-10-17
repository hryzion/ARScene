#!/usr/bin/env bash
#SBATCH --job-name=reltree-room-1000
#SBATCH --partition=general
#SBATCH --qos=advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/cscgbnu/zhx/code/ARScene
#SBATCH --output=/home/cscgbnu/zhx/code/ARScene/sampling/logs/%x-%j.out
#SBATCH --error=/home/cscgbnu/zhx/code/ARScene/sampling/logs/%x-%j.out

set -euo pipefail
RELTREE_EVAL_ROOT="${1:?evaluation root required}"
RELTREE_CHECKPOINT="${2:-networks/reltree/runs/livingroom_root_ar_12x768_attr256_cd10/last.pt}"
RELTREE_SAMPLE_DIR="${3:-$RELTREE_EVAL_ROOT/reltree_last}"
RELTREE_PYTHON=/home/cscgbnu/.conda/envs/arscene/bin/python
export PYTHONPATH="$PWD/.runtime/scenefoundry:$PWD/.runtime/diffuscene:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/arscene-room-sampling-matplotlib
mkdir -p "$RELTREE_SAMPLE_DIR"
exec 9>"$RELTREE_SAMPLE_DIR/sample.lock"
flock -n 9 || exit 1
echo "[$(date -Is)] model=reltree job=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"$RELTREE_PYTHON" -u -m sampling.sample_reltree_room_condition \
    --checkpoint "$RELTREE_CHECKPOINT" \
    --conditions "$RELTREE_EVAL_ROOT/conditions.json" --output "$RELTREE_SAMPLE_DIR" \
    --batch-size 64 --precision bf16 --seed 20261012
"$RELTREE_PYTHON" -u -m sampling.evaluate_room_condition \
    --samples "$RELTREE_SAMPLE_DIR" --conditions "$RELTREE_EVAL_ROOT/conditions.json"
