#!/bin/bash
# One stage only; no automatic sbatch, test, additional seeds or budget extension.
#SBATCH --job-name=GRID-LIGER-384MultiTrain42
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --partition=rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=12:00:00
set -euo pipefail
cd /mnt/beegfs/home/MMReco2021/liuyu/GRID
: "${SOURCE_GIT_COMMIT:?Missing source binding}"
: "${LIGER_MULTIEPOCH_MANIFEST_SHA256:?Missing manifest binding}"
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
[[ $# == 2 || $# == 3 ]] || exit 2
[[ "$2" =~ ^([1-9]|1[0-2])$ ]] || exit 2
run_bound 1 -m pytest -q tests/test_liger_early_stopping.py tests/test_liger_capacity_multiepoch_manifest.py tests/test_liger_capacity_multiepoch_training.py
if [[ $# == 3 ]]; then
 run_bound 1 -m scripts.train_liger_capacity_multiepoch run "$1" --epoch "$2" --resume-entry "$3"
else
 run_bound 1 -m scripts.train_liger_capacity_multiepoch run "$1" --epoch "$2"
fi
run_bound 1 -m scripts.train_liger_capacity_multiepoch accept "$1" --epoch "$2"
