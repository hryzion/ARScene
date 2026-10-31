#!/usr/bin/env bash
#SBATCH --job-name=reltree-dining-12x768
#SBATCH --partition=general
#SBATCH --qos=advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --chdir=/home/cscgbnu/zhx/code/ARScene
#SBATCH --output=/home/cscgbnu/zhx/code/ARScene/networks/reltree/slurm/%x-%j.out
#SBATCH --error=/home/cscgbnu/zhx/code/ARScene/networks/reltree/slurm/%x-%j.out

set -euo pipefail
exec bash networks/reltree/run_diningroom_root_train.sh
