#!/usr/bin/env bash
#SBATCH --job-name=room-condition-sample-1000
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
ROOM_SAMPLE_MODEL="${1:?model required}"
ROOM_SAMPLE_ROOT="${2:?evaluation root required}"
case "$ROOM_SAMPLE_MODEL" in
  atiss)
    ROOM_SAMPLE_PYTHON=/home/cscgbnu/.conda/envs/diffuscene/bin/python
    ROOM_SAMPLE_CHECKPOINT=pretrained/atiss/livingroom_roomshape_12x768_room64_500k/best.pt
    ROOM_SAMPLE_BATCH=1
    export PYTHONPATH="$PWD/.runtime/diffuscene:$PWD${PYTHONPATH:+:$PYTHONPATH}"
    ;;
  diffuscene)
    ROOM_SAMPLE_PYTHON=/home/cscgbnu/.conda/envs/arscene/bin/python
    ROOM_SAMPLE_CHECKPOINT=networks/diffuscene/runs/livingroom_roomshape_eps_500k/best.pt
    ROOM_SAMPLE_BATCH=128
    export PYTHONPATH="$PWD/.runtime/scenefoundry:$PWD/.runtime/diffuscene:$PWD${PYTHONPATH:+:$PYTHONPATH}"
    ;;
  *) exit 2 ;;
esac
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/arscene-room-sampling-matplotlib
mkdir -p "$ROOM_SAMPLE_ROOT/$ROOM_SAMPLE_MODEL"
exec 9>"$ROOM_SAMPLE_ROOT/$ROOM_SAMPLE_MODEL/sample.lock"
flock -n 9 || exit 1
echo "[$(date -Is)] model=$ROOM_SAMPLE_MODEL job=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"$ROOM_SAMPLE_PYTHON" -u -m sampling.sample_room_condition \
    --model "$ROOM_SAMPLE_MODEL" --checkpoint "$ROOM_SAMPLE_CHECKPOINT" \
    --conditions "$ROOM_SAMPLE_ROOT/conditions.json" --output "$ROOM_SAMPLE_ROOT/$ROOM_SAMPLE_MODEL" \
    --batch-size "$ROOM_SAMPLE_BATCH" --seed 20261012
"$ROOM_SAMPLE_PYTHON" -u -m sampling.evaluate_room_condition \
    --samples "$ROOM_SAMPLE_ROOT/$ROOM_SAMPLE_MODEL" --conditions "$ROOM_SAMPLE_ROOT/conditions.json"
