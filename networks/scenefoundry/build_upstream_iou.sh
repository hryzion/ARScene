#!/usr/bin/env bash
# Compile the original Rotated_IoU extension without editing its sources.
set -euo pipefail
: "${SLURM_JOB_ID:?Build inside a Slurm GPU allocation}"
SF_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SF_RUNTIME="$SF_PROJECT_ROOT/.runtime/scenefoundry"
SF_IOU_SOURCE="$SF_RUNTIME/Rotated_IoU"
SF_PYTHON="${SCENEFOUNDRY_PYTHON:-/home/cscgbnu/.conda/envs/arscene/bin/python}"
SF_IOU_COMMIT=3bdca6b20d981dffd773507e97f1b53641e98d0a
test "$(git -C "$SF_IOU_SOURCE" rev-parse HEAD)" = "$SF_IOU_COMMIT"
git -C "$SF_IOU_SOURCE" diff --exit-code -- cuda_op
export PYTHONPATH="$SF_RUNTIME:$SF_PROJECT_ROOT/.runtime/diffuscene:$SF_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$SF_RUNTIME/bin:$PATH"
export CUDA_HOME="${SCENEFOUNDRY_CUDA_HOME:-/usr/local/cuda-11.8}"
export MAX_JOBS=2
export TORCH_CUDA_ARCH_LIST="7.0;8.0;8.6;8.9;9.0+PTX"
cd "$SF_IOU_SOURCE/cuda_op"
"$SF_PYTHON" setup.py build_ext --build-lib "$SF_RUNTIME" --build-temp "$SF_RUNTIME/sort_vertices_build"
"$SF_PYTHON" -c 'import torch, sort_vertices; print("Original CUDA sorter:", sort_vertices.__file__); print("GPU:", torch.cuda.get_device_name())'
