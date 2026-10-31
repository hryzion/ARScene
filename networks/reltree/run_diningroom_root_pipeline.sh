#!/usr/bin/env bash
# Run in tmux; Slurm owns GPU allocations and each allocation resumes last.pt.
set -euo pipefail
RELTREE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$RELTREE_ROOT"
RELTREE_DINING_RUN="networks/reltree/runs/diningroom_root_ar_12x768_attr256_cd50_step_200k"
RELTREE_DINING_DATA="datasets/processed_reltree_diningroom_v2"
mkdir -p "$RELTREE_DINING_RUN" networks/reltree/slurm
exec 8>"$RELTREE_DINING_RUN/pipeline.lock"
flock -n 8 || { echo 'Diningroom RelTree pipeline already running'; exit 1; }
exec > >(tee -a "$RELTREE_DINING_RUN/pipeline.log") 2>&1
[[ -f "$RELTREE_DINING_DATA/READY" && -f "$RELTREE_DINING_DATA/VALIDATED" ]] || {
  echo 'Diningroom preprocessing and validation must complete first'; exit 1;
}
completed() {
  [[ -f "$RELTREE_DINING_RUN/completed.json" ]] && \
    rg -q '"step": 200000' "$RELTREE_DINING_RUN/completed.json"
}
if completed; then
  echo 'Diningroom RelTree already completed 200000 steps.'
  exit 0
fi
while true; do
  echo "[$(date -Is)] Submitting diningroom RelTree: 12x768, 200000 steps, balanced step condition dropout 0.5."
  if sbatch --parsable --wait networks/reltree/train_diningroom_root.sbatch | tee "$RELTREE_DINING_RUN/train_job.txt"; then
    if completed; then
      echo "[$(date -Is)] Diningroom RelTree completed 200000 steps."
      break
    fi
    echo 'Training exited without completing 200000 steps; inspect the log.'
    exit 1
  else
    RELTREE_DINING_EXIT=$?
  fi
  RELTREE_DINING_JOB="$(cut -d ';' -f 1 "$RELTREE_DINING_RUN/train_job.txt")"
  RELTREE_DINING_STATE="$(sacct -j "$RELTREE_DINING_JOB" -X -n -o State -P | head -n 1)"
  if [[ -f "$RELTREE_DINING_RUN/last.pt" ]] && \
     { [[ "$RELTREE_DINING_EXIT" -eq 75 ]] || [[ "$RELTREE_DINING_STATE" == TIMEOUT* || "$RELTREE_DINING_STATE" == PREEMPTED* ]]; }; then
    echo "[$(date -Is)] Resuming after $RELTREE_DINING_JOB ($RELTREE_DINING_STATE, exit $RELTREE_DINING_EXIT)."
  else
    echo "Stopped after $RELTREE_DINING_JOB ($RELTREE_DINING_STATE, exit $RELTREE_DINING_EXIT); inspect its Slurm log."
    exit "$RELTREE_DINING_EXIT"
  fi
done
