#!/usr/bin/env bash
#SBATCH --job-name=scenefoundry-verify
#SBATCH --partition=general
#SBATCH --qos=advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/cscgbnu/zhx/code/ARScene
#SBATCH --output=/home/cscgbnu/zhx/code/ARScene/pretrained/scenefoundry/slurm/%x-%j.out
#SBATCH --error=/home/cscgbnu/zhx/code/ARScene/pretrained/scenefoundry/slurm/%x-%j.out

set -euo pipefail
SCENEFOUNDRY_PYTHON="${SCENEFOUNDRY_PYTHON:-/home/cscgbnu/.conda/envs/arscene/bin/python}"
export PYTHONPATH="$PWD/.runtime/diffuscene${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/scenefoundry-matplotlib}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$SCENEFOUNDRY_PYTHON" -u -m networks.scenefoundry.verify_training \
  --config "${SCENEFOUNDRY_CONFIG:-training_configs/scenefoundry_livingroom_unconditional_eps_900k.yaml}"
