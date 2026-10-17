#!/usr/bin/env bash
# Supervise Slurm allocations; training itself always runs on a Slurm GPU.
set -euo pipefail
SCENEFOUNDRY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCENEFOUNDRY_ROOT"
SCENEFOUNDRY_PYTHON="${SCENEFOUNDRY_PYTHON:-/home/cscgbnu/.conda/envs/arscene/bin/python}"
export SCENEFOUNDRY_CONFIG="${SCENEFOUNDRY_CONFIG:-training_configs/scenefoundry_livingroom_roomshape_eps_500k.yaml}"
SCENEFOUNDRY_RUN="$("$SCENEFOUNDRY_PYTHON" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["training"]["output"])' "$SCENEFOUNDRY_CONFIG")"
mkdir -p pretrained/scenefoundry/slurm "$SCENEFOUNDRY_RUN"
exec 8>"$SCENEFOUNDRY_RUN/pipeline.lock"
flock -n 8 || { echo 'A SceneFoundry pipeline is already running'; exit 1; }
exec > >(tee -a "$SCENEFOUNDRY_RUN/pipeline.log") 2>&1
if [[ -f "$SCENEFOUNDRY_RUN/COMPLETE" ]]; then
  echo 'SceneFoundry run has already completed.'
  exit 0
fi
while true; do
  echo "[$(date -Is)] Submitting one Slurm GPU; automatically resume last.pt."
  if sbatch --parsable --wait training_scripts/scenefoundry_room_condition_500k.sbatch | tee "$SCENEFOUNDRY_RUN/train_job.txt"; then
    [[ -f "$SCENEFOUNDRY_RUN/COMPLETE" ]] || { echo 'Training exited without completing its budget'; exit 1; }
    break
  else
    SCENEFOUNDRY_EXIT=$?
  fi
  SCENEFOUNDRY_JOB="$(cut -d ';' -f 1 "$SCENEFOUNDRY_RUN/train_job.txt")"
  SCENEFOUNDRY_STATE="$(sacct -j "$SCENEFOUNDRY_JOB" -X -n -o State -P | head -n 1)"
  if [[ -f "$SCENEFOUNDRY_RUN/last.pt" ]] && \
     { [[ "$SCENEFOUNDRY_EXIT" -eq 75 ]] || [[ "$SCENEFOUNDRY_STATE" == TIMEOUT* || "$SCENEFOUNDRY_STATE" == PREEMPTED* ]]; }; then
    echo "Resuming after job=$SCENEFOUNDRY_JOB state=$SCENEFOUNDRY_STATE exit=$SCENEFOUNDRY_EXIT"
  else
    echo "Stopped: job=$SCENEFOUNDRY_JOB state=$SCENEFOUNDRY_STATE exit=$SCENEFOUNDRY_EXIT; inspect pretrained/scenefoundry/slurm"
    exit "$SCENEFOUNDRY_EXIT"
  fi
done
