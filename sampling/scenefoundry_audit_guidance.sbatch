#!/usr/bin/env bash
#SBATCH --job-name=scenefoundry-guidance-audit
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
export PYTHONPATH="$PWD/.runtime/scenefoundry:$PWD/.runtime/diffuscene${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR=/tmp/arscene-scenefoundry-sampling-matplotlib
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
/home/cscgbnu/.conda/envs/arscene/bin/python -u -m networks.scenefoundry.audit_guidance_strength \
    --guided pretrained/scenefoundry/livingroom_unconditional_eps_900k/samples_best_guidance_1000_20261009_112147 \
    --unguided pretrained/scenefoundry/livingroom_unconditional_eps_900k/samples_best_no_guidance_1000_20261009_124458 \
    --output pretrained/scenefoundry/livingroom_unconditional_eps_900k/guidance_strength_audit.json
