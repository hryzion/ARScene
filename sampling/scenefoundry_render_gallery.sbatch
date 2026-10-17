#!/usr/bin/env bash
#SBATCH --job-name=scenefoundry-original-gallery
#SBATCH --partition=general
#SBATCH --qos=advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/cscgbnu/zhx/code/ARScene
#SBATCH --output=/home/cscgbnu/zhx/code/ARScene/sampling/logs/%x-%j.out
#SBATCH --error=/home/cscgbnu/zhx/code/ARScene/sampling/logs/%x-%j.out
set -euo pipefail
export PYTHONPATH="$PWD/.runtime/scenefoundry:$PWD/.runtime/diffuscene:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR=/tmp/arscene-scenefoundry-sampling-matplotlib
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
/home/cscgbnu/.conda/envs/arscene/bin/python -u -m sampling.render_scenefoundry_gallery --samples "${1:?Pass completed sample directory}" --workers 4
