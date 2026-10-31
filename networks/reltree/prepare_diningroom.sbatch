#!/usr/bin/env bash
#SBATCH --job-name=reltree-prepare-dining
#SBATCH --partition=general
#SBATCH --qos=advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/cscgbnu/zhx/code/ARScene
#SBATCH --output=/home/cscgbnu/zhx/code/ARScene/networks/reltree/slurm/%x-%j.out
#SBATCH --error=/home/cscgbnu/zhx/code/ARScene/networks/reltree/slurm/%x-%j.out

set -euo pipefail
exec bash networks/reltree/run_prepare_diningroom.sh
