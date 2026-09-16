#!/bin/bash
# Usage: sbatch run_smoke_cpu_resume.sh TEACHER_CKPT SEED RESUME_CKPT
# Same formal launcher, 100 resumed optimizer steps and a short validation.
# Do not use this run's checkpoint or metrics as a formal experiment result.
#SBATCH --job-name=GRID-CPU-Resume-Smoke
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:3
#SBATCH --nodelist=aias-compute-01

set -euo pipefail
if [[ "$#" -ne 3 ]]; then
    echo 'Usage: sbatch run_smoke_cpu_resume.sh TEACHER_CKPT SEED RESUME_CKPT' >&2
    exit 2
fi
export GRID_RESUME_SMOKE=1
GRID_ROOT=/mnt/beegfs/home/MMReco2021/liuyu/GRID
PYTHON_BIN=/home/MMReco2021/.conda/envs/GRID/bin/python
cd "${GRID_ROOT}"
source "${GRID_ROOT}/scripts/cpu_budget.sh"
# TensorFlow-linked tests must run on the compute node, not the login CPU.
"${PYTHON_BIN}" -m pytest -q -p no:cacheprovider \
    tests/test_runtime_cpu.py tests/test_checkpoint_resume.py \
    tests/test_cpu_resume_launcher.py tests/test_cpu_resume_audit.py \
    tests/test_history_last_launchers.py tests/test_repository_launchers.py \
    tests/test_dataloading_empty_worker.py tests/test_verify_taobao_training_data.py \
    tests/test_stage_smoke.py
exec bash "${GRID_ROOT}/run_click_listwise_history_last_user_v1.sh" "$@"
