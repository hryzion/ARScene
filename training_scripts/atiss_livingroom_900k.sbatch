#!/usr/bin/env bash
#SBATCH --job-name=atiss-living-uncond-900k
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
#SBATCH --output=/home/cscgbnu/zhx/code/ARScene/pretrained/atiss/livingroom_unconditional_900k/slurm/%x-%j.out
#SBATCH --error=/home/cscgbnu/zhx/code/ARScene/pretrained/atiss/livingroom_unconditional_900k/slurm/%x-%j.out

set -euo pipefail
ATISS_PYTHON="${ATISS_PYTHON:-/home/cscgbnu/.conda/envs/diffuscene/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/arscene-atiss-matplotlib}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
exec 9>pretrained/atiss/livingroom_unconditional_900k/train.lock
flock -n 9 || { echo 'An ATISS trainer already holds this run directory'; exit 1; }
echo "[$(date -Is)] ATISS job=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
exec "$ATISS_PYTHON" -u train_atiss.py \
  --config training_configs/atiss_livingroom_unconditional_900k.yaml \
  --conditioning unconditional --max-steps 900000 --device cuda:0 --resume auto
