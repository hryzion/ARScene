#!/usr/bin/env bash
# Detached supervisor: all model computation stays inside single-GPU Slurm jobs.
set -euo pipefail
cd /home/cscgbnu/zhx/code/ARScene
ROOM_PIPELINE_MODEL="${1:?atiss or diffuscene required}"
case "$ROOM_PIPELINE_MODEL" in
  atiss) ROOM_PIPELINE_RUN=pretrained/atiss/livingroom_roomshape_12x768_room64_500k ;;
  diffuscene) ROOM_PIPELINE_RUN=networks/diffuscene/runs/livingroom_roomshape_eps_500k ;;
  *) echo 'Expected atiss or diffuscene'; exit 2 ;;
esac
mkdir -p "$ROOM_PIPELINE_RUN/slurm"
exec 8>"$ROOM_PIPELINE_RUN/pipeline.lock"
flock -n 8 || { echo 'Pipeline already running'; exit 1; }
exec > >(tee -a "$ROOM_PIPELINE_RUN/pipeline.log") 2>&1
if [[ -f "$ROOM_PIPELINE_RUN/COMPLETE" ]]; then
  echo 'Run already complete'; exit 0
fi
while true; do
  echo "[$(date -Is)] Submitting $ROOM_PIPELINE_MODEL room condition, total 500000 steps."
  if sbatch --parsable --wait --job-name="${ROOM_PIPELINE_MODEL}-room-500k" \
      --output="$PWD/$ROOM_PIPELINE_RUN/slurm/%x-%j.out" \
      --error="$PWD/$ROOM_PIPELINE_RUN/slurm/%x-%j.out" \
      training_scripts/room_condition_500k.sbatch "$ROOM_PIPELINE_MODEL" | tee "$ROOM_PIPELINE_RUN/train_job.txt"; then
    /home/cscgbnu/.conda/envs/arscene/bin/python - "$ROOM_PIPELINE_RUN" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
last = json.loads((root / 'metrics.jsonl').read_text().strip().splitlines()[-1])
assert last['event'] == 'complete' and last['step'] == 500000, last
(root / 'COMPLETE').write_text(json.dumps(last) + '\n')
PY
    echo "[$(date -Is)] Completed 500000 steps."
    break
  else
    ROOM_PIPELINE_EXIT=$?
  fi
  ROOM_PIPELINE_JOB="$(cut -d ';' -f 1 "$ROOM_PIPELINE_RUN/train_job.txt")"
  ROOM_PIPELINE_STATE="$(sacct -j "$ROOM_PIPELINE_JOB" -X -n -o State -P | head -n 1)"
  if [[ -f "$ROOM_PIPELINE_RUN/last.pt" ]] && \
      { [[ "$ROOM_PIPELINE_EXIT" -eq 75 ]] || [[ "$ROOM_PIPELINE_STATE" == TIMEOUT* || "$ROOM_PIPELINE_STATE" == PREEMPTED* ]]; }; then
    echo "Resuming job=$ROOM_PIPELINE_JOB state=$ROOM_PIPELINE_STATE exit=$ROOM_PIPELINE_EXIT."
  else
    echo "Stopped: job=$ROOM_PIPELINE_JOB state=$ROOM_PIPELINE_STATE exit=$ROOM_PIPELINE_EXIT; inspect logs."
    exit "$ROOM_PIPELINE_EXIT"
  fi
done
