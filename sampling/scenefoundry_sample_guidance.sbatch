#!/usr/bin/env bash
#SBATCH --job-name=scenefoundry-guidance-1000
#SBATCH --partition=general
#SBATCH --qos=advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/cscgbnu/zhx/code/ARScene
#SBATCH --output=/home/cscgbnu/zhx/code/ARScene/sampling/logs/%x-%j.out
#SBATCH --error=/home/cscgbnu/zhx/code/ARScene/sampling/logs/%x-%j.out

set -euo pipefail
SF_SAMPLE_RUN="${1:?Pass the output directory containing frozen checkpoint.pt}"
shift
SF_SAMPLE_PYTHON=/home/cscgbnu/.conda/envs/arscene/bin/python
export PYTHONPATH="$PWD/.runtime/scenefoundry:$PWD/.runtime/diffuscene${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR=/tmp/arscene-scenefoundry-sampling-matplotlib
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
exec 9>"$SF_SAMPLE_RUN/sample.lock"
flock -n 9 || { echo 'Sampling already running in this directory'; exit 1; }
echo "[$(date -Is)] job=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"$SF_SAMPLE_PYTHON" -u -m networks.scenefoundry.verify_upstream_guidance \
    --checkpoint "$SF_SAMPLE_RUN/checkpoint.pt" --output "$SF_SAMPLE_RUN/guidance_verification.json"
"$SF_SAMPLE_PYTHON" -u -m sampling.sample_scenefoundry \
    --checkpoint "$SF_SAMPLE_RUN/checkpoint.pt" --output "$SF_SAMPLE_RUN" \
    --count 1000 --batch-size 128 --seed 20261009 "$@"
"$SF_SAMPLE_PYTHON" -m sampling.evaluate_midiffusion --samples "$SF_SAMPLE_RUN" --expected-count 1000
