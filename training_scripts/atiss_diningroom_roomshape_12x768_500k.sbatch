#!/usr/bin/env bash
#SBATCH --job-name=atiss-dining-room-500k
#SBATCH --partition=general
#SBATCH --qos=advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --chdir=/home/cscgbnu/zhx/code/ARScene
#SBATCH --output=/home/cscgbnu/zhx/code/ARScene/pretrained/atiss/diningroom_roomshape_12x768_room64_500k/slurm/%x-%j.out
#SBATCH --error=/home/cscgbnu/zhx/code/ARScene/pretrained/atiss/diningroom_roomshape_12x768_room64_500k/slurm/%x-%j.out

set -euo pipefail
ATISS_PYTHON="${ATISS_PYTHON:-/home/cscgbnu/.conda/envs/diffuscene/bin/python}"
ATISS_CONFIG=training_configs/atiss_diningroom_roomshape_12x768_500k.yaml
ATISS_RUN=pretrained/atiss/diningroom_roomshape_12x768_room64_500k
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/arscene-atiss-matplotlib}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
exec 9>"$ATISS_RUN/train.lock"
flock -n 9 || { echo 'An ATISS trainer already holds this run directory'; exit 1; }
echo "[$(date -Is)] ATISS roomshape job=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
if [[ ! -f "$ATISS_RUN/VERIFIED" ]]; then
  "$ATISS_PYTHON" -u -m training_scripts.audit_room_masks \
    --root datasets/atiss_diningroom_obb_v2_wo_lat --output "$ATISS_RUN/mask_audit.json"
  "$ATISS_PYTHON" -u -m training_scripts.verify_room_condition \
    --model atiss --config "$ATISS_CONFIG" --mask-audit "$ATISS_RUN/mask_audit.json"
fi
exec "$ATISS_PYTHON" -u train_atiss.py \
  --config "$ATISS_CONFIG" --conditioning roomshape --max-steps 500000 --device cuda:0 --resume auto
