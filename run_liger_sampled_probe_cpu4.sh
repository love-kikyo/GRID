#!/bin/bash
# Sampled train-only resource/recovery gate; never launches formal training.
#SBATCH --job-name=GRID-LIGER-SampledV1
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
: "${LIGER_SAMPLED_MANIFEST_SHA256:?Missing manifest binding}"
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
run_bound -m pytest -q tests/test_liger_sampled_catalog.py tests/test_liger_sampled_probe_manifest.py tests/test_liger_sampled_probe.py tests/test_liger_prefix_training.py tests/test_liger_efficiency_timing.py
run_bound -m scripts.probe_liger_sampled fixture "$1"
run_bound -m scripts.probe_liger_sampled case "$1" --case n8192
run_bound -m scripts.probe_liger_sampled resume "$1" --case n8192
run_bound -m scripts.accept_liger_sampled case "$1" --case n8192
run_bound -m scripts.probe_liger_sampled case "$1" --case n32768
run_bound -m scripts.probe_liger_sampled resume "$1" --case n32768
run_bound -m scripts.accept_liger_sampled case "$1" --case n32768
run_bound -m scripts.accept_liger_sampled accept "$1"
