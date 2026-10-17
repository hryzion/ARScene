#!/usr/bin/env bash
# Resume the exact 500k update budget across Slurm time limits.
set -euo pipefail
cd /home/cscgbnu/zhx/code/ARScene
MID_ROOM_RUN=pretrained/midiffusion/livingroom_roomshape_12x768_500k
mkdir -p "$MID_ROOM_RUN/slurm"
exec 8>"$MID_ROOM_RUN/pipeline.lock"
flock -n 8 || { echo 'MiDiffusion room pipeline already running'; exit 1; }
exec > >(tee -a "$MID_ROOM_RUN/pipeline.log") 2>&1
if [[ -f "$MID_ROOM_RUN/COMPLETE" ]]; then
    echo 'MiDiffusion room training already completed'; exit 0
fi
while true; do
    echo "[$(date -Is)] Submitting MiDiffusion room condition; target 500000 steps."
    if sbatch --parsable --wait training_scripts/midiffusion_room_condition_500k.sbatch | tee "$MID_ROOM_RUN/train_job.txt"; then
        [[ -f "$MID_ROOM_RUN/COMPLETE" ]] || { echo 'Missing training completion marker'; exit 1; }
        echo "[$(date -Is)] Completed 500000 steps."
        break
    else
        MID_ROOM_EXIT=$?
    fi
    MID_ROOM_JOB="$(cut -d ';' -f 1 "$MID_ROOM_RUN/train_job.txt")"
    MID_ROOM_STATE="$(sacct -j "$MID_ROOM_JOB" -X -n -o State -P | head -n 1)"
    if [[ -f "$MID_ROOM_RUN/last.pt" ]] && \
       { [[ "$MID_ROOM_EXIT" -eq 75 ]] || [[ "$MID_ROOM_STATE" == TIMEOUT* || "$MID_ROOM_STATE" == PREEMPTED* ]]; }; then
        echo "Resuming job=$MID_ROOM_JOB state=$MID_ROOM_STATE exit=$MID_ROOM_EXIT."
    else
        echo "Stopped: job=$MID_ROOM_JOB state=$MID_ROOM_STATE exit=$MID_ROOM_EXIT; inspect Slurm log."
        exit "$MID_ROOM_EXIT"
    fi
done
