#!/usr/bin/env bash
#SBATCH --job-name=reltree-gt-root-audit
#SBATCH --partition=general
#SBATCH --qos=advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --chdir=/home/cscgbnu/zhx/code/ARScene
#SBATCH --output=/home/cscgbnu/zhx/code/ARScene/sampling/logs/%x-%j.out
#SBATCH --error=/home/cscgbnu/zhx/code/ARScene/sampling/logs/%x-%j.out
set -euo pipefail
export PYTHONPATH="$PWD/.runtime/scenefoundry:$PWD/.runtime/diffuscene:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
exec /home/cscgbnu/.conda/envs/arscene/bin/python -u -m sampling.audit_reltree_gt_root_inference \
  --checkpoint networks/reltree/runs/livingroom_root_ar_12x768_attr256_cd10/last.pt \
  --evaluation pretrained/room_condition_500k/evaluation_best_1000_20261012_232434 \
  --output "${1:-pretrained/room_condition_500k/evaluation_best_1000_20261012_232434/reltree_last_gt_room/inference_audit}" "${@:2}"
