#!/bin/bash
# Frozen-input cache and capacity only. Each case has a fresh process.
#SBATCH --job-name=GRID-LIGER-EffV4
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
: "${LIGER_EFFICIENCY_MANIFEST_SHA256:?Missing manifest binding}"
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 USE_TF=0 GRID_CPU_BUDGET=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
PYTHON_BIN=/home/MMReco2021/.conda/envs/GRID/bin/python
run_bound() {
 srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,threads --kill-on-bad-exit=1 "$PYTHON_BIN" "$@"
}
run_bound -m pytest -q tests/test_liger_feature_cache.py tests/test_liger_efficiency_timing.py tests/test_liger_efficiency_manifest.py tests/test_liger_efficiency_probe.py tests/test_liger_prefix_runtime.py tests/test_liger_prefix_training.py tests/test_liger_full_catalog.py
run_bound -m scripts.probe_liger_efficiency fixture "$1"
run_bound -m scripts.probe_liger_efficiency case "$1" --case lazy1024
run_bound -m scripts.probe_liger_efficiency case "$1" --case cached1024
run_bound -m scripts.accept_liger_efficiency equivalence "$1"
run_bound -m scripts.probe_liger_efficiency case "$1" --case cached2048
run_bound -m scripts.probe_liger_efficiency case "$1" --case cached4096
run_bound -m scripts.accept_liger_efficiency accept "$1"
