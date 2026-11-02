#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-training_configs/diffuscene_config.yaml}"
GPU_ID="${GPU_ID:-1}"
CPU_THREADS="${CPU_THREADS:-8}"
WANDB="${WANDB:-1}"
RESUME="${RESUME:-0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"
export VECLIB_MAXIMUM_THREADS="${CPU_THREADS}"

args=(python train_diffuscene.py --config "${CONFIG}" --cuda 0)

if [[ "${WANDB}" == "1" ]]; then
  args+=(--wandb)
fi

if [[ "${RESUME}" == "1" ]]; then
  args+=(--resume)
fi

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  args+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
fi

echo "[INFO] CONFIG=${CONFIG}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] CPU_THREADS=${CPU_THREADS}"
echo "[INFO] WANDB=${WANDB}"
echo "[INFO] RESUME=${RESUME}"
echo "[INFO] RESUME_CHECKPOINT=${RESUME_CHECKPOINT}"
exec "${args[@]}"
