#!/usr/bin/env bash
#SBATCH --job-name=interiorgpt-probe
#SBATCH --partition=general
#SBATCH --qos=advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --chdir=/home/cscgbnu/zhx/code/ARScene
#SBATCH --output=/home/cscgbnu/zhx/code/ARScene/pretrained/interiorGPT/livingroom_multicode_16x768_drop05_200k/slurm/probe-%j.out
#SBATCH --error=/home/cscgbnu/zhx/code/ARScene/pretrained/interiorGPT/livingroom_multicode_16x768_drop05_200k/slurm/probe-%j.out
set -euo pipefail
/home/cscgbnu/.conda/envs/arscene/bin/python - <<'PY'
import os
import torch
print('CUDA_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES'), flush=True)
print('torch', torch.__version__, 'available', torch.cuda.is_available(), flush=True)
assert torch.cuda.is_available()
print(torch.cuda.get_device_properties(0), flush=True)
x = torch.ones(32, device='cuda')
print('GPU sum', x.sum().item(), flush=True)
PY
