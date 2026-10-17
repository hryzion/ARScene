#!/usr/bin/env bash
#SBATCH --job-name=scenefoundry-b1-10
#SBATCH --partition=general
#SBATCH --qos=advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/cscgbnu/zhx/code/ARScene
#SBATCH --output=/home/cscgbnu/zhx/code/ARScene/sampling/logs/%x-%j.out
#SBATCH --error=/home/cscgbnu/zhx/code/ARScene/sampling/logs/%x-%j.out

set -euo pipefail
SF_COMPARE_RUN="${1:?Pass output directory containing frozen checkpoint.pt}"
SF_COMPARE_REFERENCE="${2:?Pass original batch128 guided directory}"
SF_COMPARE_UNGUIDED="${3:?Pass matching batch128 unguided directory}"
SF_COMPARE_PYTHON=/home/cscgbnu/.conda/envs/arscene/bin/python
export PYTHONPATH="$PWD/.runtime/scenefoundry:$PWD/.runtime/diffuscene${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR=/tmp/arscene-scenefoundry-sampling-matplotlib
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
exec 9>"$SF_COMPARE_RUN/sample.lock"
flock -n 9 || { echo 'Sampling already running in this directory'; exit 1; }
echo "[$(date -Is)] job=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"$SF_COMPARE_PYTHON" -u -m sampling.sample_scenefoundry_batch1_compare \
    --output "$SF_COMPARE_RUN" --reference "$SF_COMPARE_REFERENCE" --unguided "$SF_COMPARE_UNGUIDED" --count 10 --resume
"$SF_COMPARE_PYTHON" -m sampling.evaluate_midiffusion --samples "$SF_COMPARE_RUN" --expected-count 10
"$SF_COMPARE_PYTHON" -m sampling.render_scenefoundry_gallery --samples "$SF_COMPARE_RUN" --workers 4
"$SF_COMPARE_PYTHON" -m sampling.compare_scenefoundry_batchsizes \
    --batch1 "$SF_COMPARE_RUN" --batch128 "$SF_COMPARE_REFERENCE"
