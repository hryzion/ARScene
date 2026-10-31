#!/usr/bin/env bash
#SBATCH --job-name=reltree-step-drop-test
#SBATCH --partition=general
#SBATCH --qos=advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --chdir=/home/cscgbnu/zhx/code/ARScene
#SBATCH --output=/home/cscgbnu/zhx/code/ARScene/networks/reltree/slurm/%x-%j.out
#SBATCH --error=/home/cscgbnu/zhx/code/ARScene/networks/reltree/slurm/%x-%j.out

set -euo pipefail
export RELTREE_TEST_DEVICE=cuda:0
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
/home/cscgbnu/.conda/envs/arscene/bin/python -u -m networks.reltree.gpu_probe
exec /home/cscgbnu/.conda/envs/arscene/bin/python -u -m unittest \
  networks.reltree.tests.test_step_dropout \
  networks.reltree.tests.test_condition_dropout \
  networks.reltree.tests.test_room_root \
  networks.reltree.tests.test_resume -v
