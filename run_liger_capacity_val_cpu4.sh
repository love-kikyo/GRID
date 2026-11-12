#!/bin/bash
# Two catalog shards, still one task and FOUR CPUs for the WHOLE job.
#SBATCH --job-name=GRID-LIGER-Cap384Val
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --partition=rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=24G
#SBATCH --time=00:45:00
set -euo pipefail
[[ $# == 1 ]] || exit 2
cd /mnt/beegfs/home/MMReco2021/liuyu/GRID
: "${SOURCE_GIT_COMMIT:?Missing source binding}"
: "${LIGER_CAPACITY_MANIFEST_SHA256:?Missing manifest binding}"
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 USE_TF=0 GRID_CPU_BUDGET=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled NUMPY_MADVISE_HUGEPAGE=0
PYTHON_BIN=/home/MMReco2021/.conda/envs/GRID/bin/python
run_bound() {
 srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,threads --kill-on-bad-exit=1 "$PYTHON_BIN" "$@"
}
run_bound -m pytest -q tests/test_liger_sharded_retrieval.py tests/test_liger_capacity_probe.py tests/test_liger_capacity_validation_probe.py
run_bound -m scripts.probe_liger_capacity_validation "$1"
