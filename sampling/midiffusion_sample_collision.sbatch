#!/usr/bin/env bash
#SBATCH --job-name=midiffusion-sample-collision
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
MID_SAMPLE_RUN="${1:?Pass the output directory containing checkpoint.pt}"
MID_SAMPLE_PYTHON=/home/cscgbnu/.conda/envs/diffuscene/bin/python
export MPLCONFIGDIR=/tmp/arscene-midiffusion-sampling-matplotlib
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
exec 9>"$MID_SAMPLE_RUN/sample.lock"
flock -n 9 || { echo 'Sampling already running in this output directory'; exit 1; }
echo "[$(date -Is)] job=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"$MID_SAMPLE_PYTHON" -u -m sampling.sample_midiffusion \
    --checkpoint "$MID_SAMPLE_RUN/checkpoint.pt" --output "$MID_SAMPLE_RUN" \
    --count 1000 --batch-size 128 --seed 20261008
"$MID_SAMPLE_PYTHON" -m sampling.evaluate_midiffusion --samples "$MID_SAMPLE_RUN" --expected-count 1000
