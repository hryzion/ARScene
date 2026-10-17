#!/usr/bin/env bash
set -euo pipefail
export DIFFUSCENE_CONFIG=training_configs/diffuscene_livingroom_uncond_900k_aug.yaml
export DIFFUSCENE_RUN=networks/diffuscene/runs/livingroom_uncond_900k_aug
export DIFFUSCENE_JOB_NAME=diffuscene-living-900k
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_livingroom_pipeline.sh"
