#!/usr/bin/env bash
#SBATCH --job-name=diffuscene-dining-uncond-500k
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
#SBATCH --output=/home/cscgbnu/zhx/code/ARScene/networks/diffuscene/runs/diningroom_uncond_500k_aug/slurm/%x-%j.out
#SBATCH --error=/home/cscgbnu/zhx/code/ARScene/networks/diffuscene/runs/diningroom_uncond_500k_aug/slurm/%x-%j.out

set -euo pipefail
export DIFFUSCENE_CONFIG=training_configs/diffuscene_diningroom_uncond_500k_aug.yaml
export PYTHONUNBUFFERED=1
DIFFUSCENE_RUN=networks/diffuscene/runs/diningroom_uncond_500k_aug
exec 9>"$DIFFUSCENE_RUN/train.lock"
flock -n 9 || { echo 'A DiffuScene trainer already holds this run directory'; exit 1; }
echo "[$(date -Is)] DiffuScene job=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
# Preserve the previous pipeline's GPU integration check, in a separate test output.
if [[ ! -f "$DIFFUSCENE_RUN/VERIFIED" ]]; then
  bash networks/diffuscene/run_verify_livingroom.sh
fi
[[ -f "$DIFFUSCENE_RUN/VERIFIED" ]] || { echo 'GPU integration verification did not finish'; exit 1; }
exec bash networks/diffuscene/run_livingroom_train.sh
