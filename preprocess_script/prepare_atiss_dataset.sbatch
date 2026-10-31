#!/usr/bin/env bash
#SBATCH --job-name=prepare-atiss
#SBATCH --partition=general
#SBATCH --qos=advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/cscgbnu/zhx/code/ARScene
#SBATCH --output=/home/cscgbnu/zhx/code/ARScene/preprocess_script/logs/%x-%j.out
#SBATCH --error=/home/cscgbnu/zhx/code/ARScene/preprocess_script/logs/%x-%j.out

set -euo pipefail
export PYTHONPATH="$PWD/.runtime/scenefoundry:$PWD/.runtime/diffuscene:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="/tmp/arscene-preprocess-matplotlib-${SLURM_JOB_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
exec /home/cscgbnu/.conda/envs/arscene/bin/python -u -m preprocess_script.prepare_atiss_dataset "$@"
