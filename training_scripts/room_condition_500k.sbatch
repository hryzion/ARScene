#!/usr/bin/env bash
#SBATCH --job-name=room-condition-500k
#SBATCH --partition=general
#SBATCH --qos=advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --chdir=/home/cscgbnu/zhx/code/ARScene
#SBATCH --output=/home/cscgbnu/zhx/code/ARScene/training_scripts/logs/%x-%j.out
#SBATCH --error=/home/cscgbnu/zhx/code/ARScene/training_scripts/logs/%x-%j.out

set -euo pipefail
ROOM_TRAIN_MODEL="${1:?atiss or diffuscene required}"
case "$ROOM_TRAIN_MODEL" in
  atiss)
    ROOM_TRAIN_PYTHON=/home/cscgbnu/.conda/envs/diffuscene/bin/python
    ROOM_TRAIN_CONFIG=training_configs/atiss_livingroom_roomshape_12x768_500k.yaml
    ROOM_TRAIN_OUTPUT=pretrained/atiss/livingroom_roomshape_12x768_room64_500k
    ;;
  diffuscene)
    ROOM_TRAIN_PYTHON=/home/cscgbnu/.conda/envs/arscene/bin/python
    ROOM_TRAIN_CONFIG=training_configs/diffuscene_livingroom_roomshape_eps_500k.yaml
    ROOM_TRAIN_OUTPUT=networks/diffuscene/runs/livingroom_roomshape_eps_500k
    ;;
  *) echo 'Expected atiss or diffuscene'; exit 2 ;;
esac
export PYTHONPATH="$PWD/.runtime/diffuscene:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR=/tmp/arscene-room-condition-matplotlib
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
exec 9>"$ROOM_TRAIN_OUTPUT/train.lock"
flock -n 9 || { echo 'A trainer already holds this output directory'; exit 1; }
echo "[$(date -Is)] model=$ROOM_TRAIN_MODEL job=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
if [[ ! -f "$ROOM_TRAIN_OUTPUT/VERIFIED" ]]; then
  "$ROOM_TRAIN_PYTHON" -u -m training_scripts.verify_room_condition --model "$ROOM_TRAIN_MODEL" --config "$ROOM_TRAIN_CONFIG"
fi
if [[ "$ROOM_TRAIN_MODEL" == atiss ]]; then
  exec "$ROOM_TRAIN_PYTHON" -u train_atiss.py --config "$ROOM_TRAIN_CONFIG" \
    --conditioning roomshape --max-steps 500000 --device cuda:0 --resume auto
else
  exec "$ROOM_TRAIN_PYTHON" -u train_diffuscene.py --config "$ROOM_TRAIN_CONFIG" --cuda 0
fi
