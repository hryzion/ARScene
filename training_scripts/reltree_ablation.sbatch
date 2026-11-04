#!/usr/bin/env bash
#SBATCH --partition=shuye
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@180
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this file with sbatch}"
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
RELTREE_PYTHON="${RELTREE_PYTHON:-/mnt/atlas/cv/hxzhu/.venvs/arscene-preprocess-py311/bin/python}"
exec "$RELTREE_PYTHON" -B -m networks.reltree.ablations.training --config "${1:?config path required}" --resume auto
