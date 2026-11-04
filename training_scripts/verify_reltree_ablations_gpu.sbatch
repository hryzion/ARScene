#!/usr/bin/env bash
#SBATCH --partition=shuye
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
RELTREE_PYTHON="${RELTREE_PYTHON:-/mnt/atlas/cv/hxzhu/.venvs/arscene-preprocess-py311/bin/python}"
exec "$RELTREE_PYTHON" -B -m networks.reltree.ablations.verify_gpu \
  --source-root "${1:?continuous source root required}" --output "${2:?fresh verification output required}"
