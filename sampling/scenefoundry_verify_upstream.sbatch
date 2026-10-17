#!/usr/bin/env bash
#SBATCH --job-name=scenefoundry-original-verify
#SBATCH --partition=general
#SBATCH --qos=advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/cscgbnu/zhx/code/ARScene
#SBATCH --output=/home/cscgbnu/zhx/code/ARScene/sampling/logs/%x-%j.out
#SBATCH --error=/home/cscgbnu/zhx/code/ARScene/sampling/logs/%x-%j.out
set -euo pipefail
SF_CHECKPOINT="${1:?Pass frozen checkpoint.pt}"
SF_VERIFY_RUN="${2:?Pass a new verification output directory}"
SF_PYTHON=/home/cscgbnu/.conda/envs/arscene/bin/python
export PYTHONPATH="$PWD/.runtime/scenefoundry:$PWD/.runtime/diffuscene:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR=/tmp/arscene-scenefoundry-sampling-matplotlib
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mkdir -p "$SF_VERIFY_RUN"
"$SF_PYTHON" -u -m networks.scenefoundry.verify_upstream_guidance \
    --checkpoint "$SF_CHECKPOINT" --output "$SF_VERIFY_RUN/verification.json"
"$SF_PYTHON" -u -m sampling.sample_scenefoundry \
    --checkpoint "$SF_CHECKPOINT" --output "$SF_VERIFY_RUN/samples" \
    --count 4 --batch-size 4 --seed 20261009 --guidance collision
"$SF_PYTHON" -m sampling.evaluate_midiffusion --samples "$SF_VERIFY_RUN/samples" --expected-count 4
