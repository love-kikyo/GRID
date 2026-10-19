#!/bin/bash
# Real-train batch capacity only, no formal checkpoint or automatic successor.
#SBATCH --job-name=GRID-LIGER-Batch
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --partition=rtx
#SBATCH --nodelist=aias-compute-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:45:00
set -euo pipefail
[[ $# == 1 ]] || exit 2
cd /mnt/beegfs/home/MMReco2021/liuyu/GRID
: "${SOURCE_GIT_COMMIT:?Missing source binding}"
: "${LIGER_BATCH_MANIFEST_SHA256:?Missing manifest binding}"
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 USE_TF=0
export GRID_CPU_BUDGET=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
PYTHON_BIN=/home/MMReco2021/.conda/envs/GRID/bin/python
srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,threads --kill-on-bad-exit=1 \
 "$PYTHON_BIN" -m pytest -q tests/test_liger_batch_probe.py tests/test_liger_batch_launcher.py tests/test_liger_full_catalog.py tests/test_liger_inputs.py tests/test_liger_projection.py
srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,threads --kill-on-bad-exit=1 \
 "$PYTHON_BIN" -m scripts.probe_liger_batch_scaling run "$1"
srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,threads --kill-on-bad-exit=1 \
 "$PYTHON_BIN" -m scripts.probe_liger_batch_scaling accept "$1"
