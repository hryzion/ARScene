#!/usr/bin/env bash
#SBATCH --job-name=interiorgpt-sample-1000
#SBATCH --partition=general
#SBATCH --qos=advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/cscgbnu/zhx/code/ARScene
#SBATCH --output=/home/cscgbnu/zhx/code/ARScene/sampling/logs/%x-%j.out
#SBATCH --error=/home/cscgbnu/zhx/code/ARScene/sampling/logs/%x-%j.out
set -euo pipefail
INTERIORGPT_MODE="${1:?unconditional or roomconditional required}"
INTERIORGPT_ROOT="${2:?output root required}"
INTERIORGPT_COUNT="${3:-1000}"
INTERIORGPT_BATCH="${4:-32}"
INTERIORGPT_PYTHON=/home/cscgbnu/.conda/envs/arscene/bin/python
export PYTHONPATH="$PWD/.runtime/scenefoundry:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/arscene-interiorgpt-sampling-matplotlib
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$INTERIORGPT_ROOT/$INTERIORGPT_MODE"
exec 9>"$INTERIORGPT_ROOT/$INTERIORGPT_MODE/sample.lock"
flock -n 9 || exit 1
"$INTERIORGPT_PYTHON" -u -m sampling.sample_interiorgpt \
    --checkpoint pretrained/interiorGPT/livingroom_multicode_12x768_drop05_200k/last.pt \
    --mode "$INTERIORGPT_MODE" --output "$INTERIORGPT_ROOT/$INTERIORGPT_MODE" \
    --conditions "$INTERIORGPT_ROOT/conditions.json" --count "$INTERIORGPT_COUNT" \
    --batch-size "$INTERIORGPT_BATCH" --seed 20261022
