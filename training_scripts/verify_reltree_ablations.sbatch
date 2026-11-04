#!/usr/bin/env bash
#SBATCH --partition=shuye
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export RELTREE_TEST_DEVICE=cpu
RELTREE_PYTHON="${RELTREE_PYTHON:-/mnt/atlas/cv/hxzhu/.venvs/arscene-preprocess-py311/bin/python}"
"$RELTREE_PYTHON" -B -m unittest discover -s networks/reltree/ablations/tests -t . -v
"$RELTREE_PYTHON" -B -m unittest discover -s networks/reltree/tests -v
