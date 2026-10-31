#!/usr/bin/env bash
#SBATCH --job-name=interiorgpt-dining-uncond-200k
#SBATCH --partition=general
#SBATCH --qos=advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --chdir=/home/cscgbnu/zhx/code/ARScene
#SBATCH --output=/home/cscgbnu/zhx/code/ARScene/pretrained/interiorGPT/diningroom_unconditional_multicode_12x768_200k/slurm/%x-%j.out
#SBATCH --error=/home/cscgbnu/zhx/code/ARScene/pretrained/interiorGPT/diningroom_unconditional_multicode_12x768_200k/slurm/%x-%j.out

set -euo pipefail
INTERIORGPT_PYTHON="${INTERIORGPT_PYTHON:-/home/cscgbnu/.conda/envs/arscene/bin/python}"
INTERIORGPT_CONFIG=training_configs/interiorgpt_diningroom_unconditional_12x768_200k.yaml
INTERIORGPT_OUTPUT=pretrained/interiorGPT/diningroom_unconditional_multicode_12x768_200k
export MPLCONFIGDIR=/tmp/arscene-interiorgpt-matplotlib
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec 9>"$INTERIORGPT_OUTPUT/train.lock"
flock -n 9 || { echo 'An InteriorGPT trainer already holds this run directory'; exit 1; }
echo "[$(date -Is)] InteriorGPT job=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
exec "$INTERIORGPT_PYTHON" -u train_ar.py --config "$INTERIORGPT_CONFIG" --cuda 0 --resume auto
