#!/usr/bin/env bash
# tmux supervises Slurm; all Python training runs inside the GPU allocation.
set -euo pipefail
RELTREE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$RELTREE_ROOT"
RELTREE_LIVING_RUN="networks/reltree/runs/livingroom_root_ar_12x768_attr256_cd10"
RELTREE_LIVING_DATA="datasets/processed_reltree_livingroom_v2"
mkdir -p "$RELTREE_LIVING_RUN" networks/reltree/slurm
exec 8>"$RELTREE_LIVING_RUN/pipeline.lock"
flock -n 8 || { echo 'Livingroom RelTree pipeline already running'; exit 1; }
exec > >(tee -a "$RELTREE_LIVING_RUN/pipeline.log") 2>&1
[[ -f "$RELTREE_LIVING_DATA/READY" && -f "$RELTREE_LIVING_DATA/VALIDATED" ]] || {
  echo 'Livingroom preprocessing and validation must complete first'; exit 1;
}
if [[ -f "$RELTREE_LIVING_RUN/completed.json" ]] && \
   rg -q '"step": 100000' "$RELTREE_LIVING_RUN/completed.json"; then
  echo 'Livingroom RelTree already completed 100000 steps.'
  exit 0
fi
while true; do
  echo "[$(date -Is)] Submitting livingroom RelTree: one GPU, 48 hours, 100000 steps, automatic last.pt recovery."
  if sbatch --parsable --wait networks/reltree/train_livingroom_root.slurm | tee "$RELTREE_LIVING_RUN/train_job.txt"; then
    if [[ -f "$RELTREE_LIVING_RUN/completed.json" ]] && \
       rg -q '"step": 100000' "$RELTREE_LIVING_RUN/completed.json"; then
      echo "[$(date -Is)] Livingroom RelTree completed 100000 steps."
      break
    fi
    echo 'Training exited without completing 100000 steps; inspect the log.'
    exit 1
  else
    RELTREE_LIVING_EXIT=$?
  fi
  RELTREE_LIVING_JOB="$(cut -d ';' -f 1 "$RELTREE_LIVING_RUN/train_job.txt")"
  RELTREE_LIVING_STATE="$(sacct -j "$RELTREE_LIVING_JOB" -X -n -o State -P | head -n 1)"
  if [[ -f "$RELTREE_LIVING_RUN/last.pt" ]] && \
     { [[ "$RELTREE_LIVING_EXIT" -eq 75 ]] || [[ "$RELTREE_LIVING_STATE" == TIMEOUT* || "$RELTREE_LIVING_STATE" == PREEMPTED* ]]; }; then
    echo "[$(date -Is)] Resuming after $RELTREE_LIVING_JOB ($RELTREE_LIVING_STATE, exit $RELTREE_LIVING_EXIT)."
  else
    echo "Stopped after $RELTREE_LIVING_JOB ($RELTREE_LIVING_STATE, exit $RELTREE_LIVING_EXIT); inspect its Slurm log."
    exit "$RELTREE_LIVING_EXIT"
  fi
done
