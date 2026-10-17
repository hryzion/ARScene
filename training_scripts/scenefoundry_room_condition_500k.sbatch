#!/usr/bin/env bash
#SBATCH --job-name=scenefoundry-room-eps-500k
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
#SBATCH --output=/home/cscgbnu/zhx/code/ARScene/pretrained/scenefoundry/slurm/%x-%j.out
#SBATCH --error=/home/cscgbnu/zhx/code/ARScene/pretrained/scenefoundry/slurm/%x-%j.out

set -euo pipefail
SCENEFOUNDRY_PYTHON="${SCENEFOUNDRY_PYTHON:-/home/cscgbnu/.conda/envs/arscene/bin/python}"
SCENEFOUNDRY_CONFIG="${SCENEFOUNDRY_CONFIG:-training_configs/scenefoundry_livingroom_roomshape_eps_500k.yaml}"
export PYTHONPATH="$PWD/.runtime/diffuscene${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/scenefoundry-matplotlib}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
SCENEFOUNDRY_RUN="$("$SCENEFOUNDRY_PYTHON" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["training"]["output"])' "$SCENEFOUNDRY_CONFIG")"
mkdir -p "$SCENEFOUNDRY_RUN"
exec 9>"$SCENEFOUNDRY_RUN/train.lock"
flock -n 9 || { echo 'A SceneFoundry trainer already holds this run directory'; exit 1; }
echo "[$(date -Is)] SceneFoundry job=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
exec "$SCENEFOUNDRY_PYTHON" -u train_scenefoundry.py --config "$SCENEFOUNDRY_CONFIG" --cuda 0 --resume auto
