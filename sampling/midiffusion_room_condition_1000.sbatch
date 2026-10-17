#!/usr/bin/env bash
#SBATCH --job-name=midiffusion-room-1000
#SBATCH --partition=general
#SBATCH --qos=advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/cscgbnu/zhx/code/ARScene
#SBATCH --output=/home/cscgbnu/zhx/code/ARScene/sampling/logs/%x-%j.out
#SBATCH --error=/home/cscgbnu/zhx/code/ARScene/sampling/logs/%x-%j.out

set -euo pipefail
MID_SAMPLE_ROOT="${1:?evaluation root required}"
MID_SAMPLE_PYTHON=/home/cscgbnu/.conda/envs/arscene/bin/python
export PYTHONPATH="$PWD/.runtime/scenefoundry:$PWD/.runtime/diffuscene:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/arscene-room-sampling-matplotlib
mkdir -p "$MID_SAMPLE_ROOT/midiffusion"
exec 9>"$MID_SAMPLE_ROOT/midiffusion/sample.lock"
flock -n 9 || exit 1
echo "[$(date -Is)] model=midiffusion job=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"$MID_SAMPLE_PYTHON" -u -m sampling.sample_midiffusion_room_condition \
    --checkpoint pretrained/midiffusion/livingroom_roomshape_12x768_500k/best.pt \
    --conditions "$MID_SAMPLE_ROOT/conditions.json" --output "$MID_SAMPLE_ROOT/midiffusion" \
    --batch-size 128 --seed 20261012
"$MID_SAMPLE_PYTHON" -u -m sampling.evaluate_room_condition \
    --samples "$MID_SAMPLE_ROOT/midiffusion" --conditions "$MID_SAMPLE_ROOT/conditions.json"
