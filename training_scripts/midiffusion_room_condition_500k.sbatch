#!/usr/bin/env bash
#SBATCH --job-name=midiffusion-room-12x768-500k
#SBATCH --partition=general
#SBATCH --qos=advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --chdir=/home/cscgbnu/zhx/code/ARScene
#SBATCH --output=/home/cscgbnu/zhx/code/ARScene/pretrained/midiffusion/livingroom_roomshape_12x768_500k/slurm/%x-%j.out
#SBATCH --error=/home/cscgbnu/zhx/code/ARScene/pretrained/midiffusion/livingroom_roomshape_12x768_500k/slurm/%x-%j.out

set -euo pipefail
MID_ROOM_PYTHON=/home/cscgbnu/.conda/envs/arscene/bin/python
MID_ROOM_CONFIG=training_configs/midiffusion_livingroom_roomshape_12x768_500k.yaml
MID_ROOM_OUTPUT=pretrained/midiffusion/livingroom_roomshape_12x768_500k
export PYTHONPATH="$PWD/.runtime/scenefoundry:$PWD/.runtime/diffuscene:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR=/tmp/arscene-midiffusion-room-matplotlib
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
exec 9>"$MID_ROOM_OUTPUT/train.lock"
flock -n 9 || { echo 'A trainer already holds this run directory'; exit 1; }
echo "[$(date -Is)] MiDiffusion room condition job=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
if [[ ! -f "$MID_ROOM_OUTPUT/VERIFIED" ]]; then
    "$MID_ROOM_PYTHON" -u -m training_scripts.verify_midiffusion_room_condition --config "$MID_ROOM_CONFIG"
fi
exec "$MID_ROOM_PYTHON" -u train_midiffusion.py --config "$MID_ROOM_CONFIG" --cuda 0
