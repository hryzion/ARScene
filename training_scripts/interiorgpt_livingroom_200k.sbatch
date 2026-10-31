#!/usr/bin/env bash
#SBATCH --job-name=interiorgpt-12x768-200k
#SBATCH --partition=general
#SBATCH --qos=advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --chdir=/home/cscgbnu/zhx/code/ARScene
#SBATCH --output=/home/cscgbnu/zhx/code/ARScene/pretrained/interiorGPT/livingroom_multicode_12x768_drop05_200k/slurm/%x-%j.out
#SBATCH --error=/home/cscgbnu/zhx/code/ARScene/pretrained/interiorGPT/livingroom_multicode_12x768_drop05_200k/slurm/%x-%j.out

set -euo pipefail
INTERIORGPT_PYTHON=/home/cscgbnu/.conda/envs/arscene/bin/python
INTERIORGPT_CONFIG=training_configs/interiorgpt_livingroom_multicode_200k.yaml
INTERIORGPT_OUTPUT=pretrained/interiorGPT/livingroom_multicode_12x768_drop05_200k
export MPLCONFIGDIR=/tmp/arscene-interiorgpt-matplotlib
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec 9>"$INTERIORGPT_OUTPUT/train.lock"
flock -n 9 || { echo 'An InteriorGPT trainer already holds this run directory'; exit 1; }
echo "[$(date -Is)] InteriorGPT job=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "$SLURM_JOB_ID" > "$INTERIORGPT_OUTPUT/train_job.txt"

# Forward Slurm's advance time-limit notice. The trainer saves its consumed-sample
# cursor after the current optimizer update and returns 75 for a resumable stop.
INTERIORGPT_PID=''
trap '[[ -z "$INTERIORGPT_PID" ]] || kill -USR1 "$INTERIORGPT_PID" 2>/dev/null || true' USR1
trap '[[ -z "$INTERIORGPT_PID" ]] || kill -TERM "$INTERIORGPT_PID" 2>/dev/null || true' TERM INT
"$INTERIORGPT_PYTHON" -u train_ar.py --config "$INTERIORGPT_CONFIG" --cuda 0 --resume auto &
INTERIORGPT_PID=$!
while true; do
    if wait "$INTERIORGPT_PID"; then INTERIORGPT_EXIT=0; else INTERIORGPT_EXIT=$?; fi
    kill -0 "$INTERIORGPT_PID" 2>/dev/null || break
done
if [[ "$INTERIORGPT_EXIT" -eq 75 && -f "$INTERIORGPT_OUTPUT/last.pt" ]]; then
    sbatch --parsable --dependency="afterany:$SLURM_JOB_ID" \
        training_scripts/interiorgpt_livingroom_200k.sbatch > "$INTERIORGPT_OUTPUT/next_job.txt"
    echo "Time limit approaching; queued continuation $(cat "$INTERIORGPT_OUTPUT/next_job.txt")"
    exit 0
fi
exit "$INTERIORGPT_EXIT"
