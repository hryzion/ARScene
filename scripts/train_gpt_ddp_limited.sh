#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-training_configs/gpt_config_living.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CPU_THREADS="${CPU_THREADS:-4}"
WANDB="${WANDB:-1}"
RESUME="${RESUME:-0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
MASTER_PORT="${MASTER_PORT:-29501}"

export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"
export VECLIB_MAXIMUM_THREADS="${CPU_THREADS}"

args=(torchrun --standalone --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" train_gpt.py --config "${CONFIG}")

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
echo "[INFO] NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "[INFO] CPU_THREADS=${CPU_THREADS}"
echo "[INFO] WANDB=${WANDB}"
echo "[INFO] RESUME=${RESUME}"
echo "[INFO] MASTER_PORT=${MASTER_PORT}"
exec "${args[@]}"
