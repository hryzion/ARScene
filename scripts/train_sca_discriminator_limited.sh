#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ROOT="${ROOT:-./generated}"
GPU_ID="${GPU_ID:-0}"
CPU_THREADS="${CPU_THREADS:-8}"
WORKERS="${WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-./pretrained/sca_discriminator}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"
export VECLIB_MAXIMUM_THREADS="${CPU_THREADS}"

args=(
  python train_sca_discriminator.py
  --root "${ROOT}"
  --cuda 0
  --workers "${WORKERS}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --output-dir "${OUTPUT_DIR}"
)

echo "[INFO] ROOT=${ROOT}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] CPU_THREADS=${CPU_THREADS}"
echo "[INFO] WORKERS=${WORKERS}"
echo "[INFO] BATCH_SIZE=${BATCH_SIZE}"
echo "[INFO] EPOCHS=${EPOCHS}"
exec "${args[@]}"
