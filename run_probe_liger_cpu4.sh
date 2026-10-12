#!/bin/bash
# Compatibility-only official LIGER probe; does not start formal training.
#SBATCH --job-name=GRID-LIGER-Probe
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --partition=rtx
#SBATCH --nodelist=aias-compute-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:20:00
set -euo pipefail
[[ "$#" == 1 ]] || { echo 'Usage: sbatch run_probe_liger_cpu4.sh MANIFEST' >&2; exit 2; }
cd /mnt/beegfs/home/MMReco2021/liuyu/GRID
: "${SOURCE_GIT_COMMIT:?Missing source binding}"
: "${BASELINE_MANIFEST_SHA256:?Missing manifest binding}"
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export GRID_CPU_BUDGET=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE USE_TF=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
PYTHON_BIN=/home/MMReco2021/.conda/envs/GRID/bin/python
srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,threads --kill-on-bad-exit=1 \
    "$PYTHON_BIN" -m pytest -q tests/test_liger_inputs.py tests/test_liger_probe.py tests/test_export_cpu_budget.py tests/test_repository_launchers.py
srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,threads --kill-on-bad-exit=1 \
    "$PYTHON_BIN" -m scripts.probe_liger_cpu4 --run "$1"
