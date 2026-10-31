#!/usr/bin/env bash
#SBATCH --job-name=scenefoundry-room-guidance-1000
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
SF_ROOM_ROOT="${1:?evaluation root required}"
SF_ROOM_OUTPUT="${2:-$SF_ROOM_ROOT/scenefoundry_collision_room_guidance}"
SF_ROOM_GUIDANCE="${3:-collision_room}"
SF_ROOM_PYTHON=/home/cscgbnu/.conda/envs/arscene/bin/python
export PYTHONPATH="$PWD/.runtime/scenefoundry:$PWD/.runtime/diffuscene:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/arscene-scenefoundry-room-sampling-matplotlib
mkdir -p "$SF_ROOM_OUTPUT"
exec 9>"$SF_ROOM_OUTPUT/sample.lock"
flock -n 9 || { echo 'SceneFoundry room sampling already running'; exit 1; }
echo "[$(date -Is)] model=scenefoundry job=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"$SF_ROOM_PYTHON" -u -m sampling.sample_scenefoundry_room_condition \
    --checkpoint pretrained/scenefoundry/livingroom_roomshape_eps_500k/best.pt \
    --conditions "$SF_ROOM_ROOT/conditions.json" --output "$SF_ROOM_OUTPUT" \
    --guidance "$SF_ROOM_GUIDANCE" --batch-size 128 --seed 20261012
"$SF_ROOM_PYTHON" -u -m sampling.evaluate_room_condition \
    --samples "$SF_ROOM_OUTPUT" --conditions "$SF_ROOM_ROOT/conditions.json"
"$SF_ROOM_PYTHON" -u -c 'import sys; from pathlib import Path; from sampling.summarize_room_condition import audit_decode; print(audit_decode(Path(sys.argv[1])))' "$SF_ROOM_OUTPUT"
