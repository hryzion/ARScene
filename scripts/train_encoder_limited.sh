#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-training_configs/residual_vq_config_living_wo_lat.yaml}"
GPU_ID="${GPU_ID:-1}"
CPU_THREADS="${CPU_THREADS:-8}"
WANDB="${WANDB:-1}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"
export VECLIB_MAXIMUM_THREADS="${CPU_THREADS}"

args=(python train_encoder.py --config "${CONFIG}" --cuda 0)

if [[ "${WANDB}" == "1" ]]; then
  args+=(--wandb)
fi

echo "[INFO] CONFIG=${CONFIG}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] CPU_THREADS=${CPU_THREADS}"
echo "[INFO] WANDB=${WANDB}"
exec "${args[@]}"
