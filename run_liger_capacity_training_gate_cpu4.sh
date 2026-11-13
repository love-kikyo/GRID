#!/bin/bash
# One authorized pass only; no test, new seed, epoch2 or automatic submission.
#SBATCH --job-name=GRID-LIGER-384Gate
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
: "${LIGER_CAPACITY_TRAIN_MANIFEST_SHA256:?Missing training manifest binding}"
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 USE_TF=0 GRID_CPU_BUDGET=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled NUMPY_MADVISE_HUGEPAGE=0
PYTHON_BIN=/home/MMReco2021/.conda/envs/GRID/bin/python
run_bound() {
 local allocated_gpus="$1"
 shift
 srun --ntasks=1 --cpus-per-task=4 --gres=gpu:"$allocated_gpus" --cpu-bind=verbose,threads --kill-on-bad-exit=1 "$PYTHON_BIN" "$@"
}
run_bound 1 -m pytest -q tests/test_liger_capacity_training_manifest.py tests/test_liger_capacity_training_entrypoints.py tests/test_liger_capacity_full_validation.py tests/test_liger_capacity_training_gate.py
run_bound 1 -m scripts.probe_liger_capacity_training continuous "$1"
run_bound 1 -m scripts.probe_liger_capacity_training resumed "$1"
run_bound 1 -m scripts.probe_liger_capacity_training accept_train "$1"
run_bound 2 -m scripts.probe_liger_capacity_training validation "$1"
run_bound 2 -m scripts.probe_liger_capacity_training accept "$1"
