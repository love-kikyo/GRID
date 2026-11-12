#!/bin/bash
# One bounded train-only cache/read/capacity probe, no formal successor.
#SBATCH --job-name=GRID-LIGER-ScalingV3
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --partition=rtx4090
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
: "${LIGER_SCALING_MANIFEST_SHA256:?Missing manifest binding}"
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 USE_TF=0 GRID_CPU_BUDGET=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
# Process-local allocation policy only; no kernel huge-page settings are changed.
export NUMPY_MADVISE_HUGEPAGE=0
PYTHON_BIN=/home/MMReco2021/.conda/envs/GRID/bin/python
run_bound() {
 srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,threads --kill-on-bad-exit=1 "$PYTHON_BIN" "$@"
}
run_bound -m pytest -q tests/test_liger_access_timing.py tests/test_liger_row_lookup.py tests/test_liger_sampled_scaling_manifest.py tests/test_liger_sampled_scaling.py tests/test_liger_feature_cache.py tests/test_liger_sampled_catalog.py tests/test_liger_prefix_training.py
run_bound -m scripts.probe_liger_sampled_scaling fixture "$1"
run_bound -m scripts.probe_liger_sampled_scaling case "$1" --case lazy8k1024
run_bound -m scripts.probe_liger_sampled_scaling case "$1" --case cached8k1024
run_bound -m scripts.probe_liger_sampled_scaling case "$1" --case cached32k1024
run_bound -m scripts.probe_liger_sampled_scaling case "$1" --case lazy32k1024
run_bound -m scripts.accept_liger_sampled_scaling equivalence "$1"
run_bound -m scripts.probe_liger_sampled_scaling case "$1" --case cached8k2048
run_bound -m scripts.probe_liger_sampled_scaling case "$1" --case cached32k2048
run_bound -m scripts.probe_liger_sampled_scaling case "$1" --case cached8k4096
run_bound -m scripts.probe_liger_sampled_scaling case "$1" --case cached32k4096
run_bound -m scripts.accept_liger_sampled_scaling accept "$1"
