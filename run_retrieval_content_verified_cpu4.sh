#!/bin/bash
#SBATCH --job-name=RetrContent42-Full
#SBATCH --output=/mnt/beegfs/home/MMReco2021/liuyu/GRID/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/beegfs/home/MMReco2021/liuyu/GRID/logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:4090:1
#SBATCH --partition=rtx4090
#SBATCH --nodelist=aias-compute-2
#SBATCH --time=06:00:00
set -euo pipefail
if [[ $# -ne 1 || "$1" != /* ]]; then echo 'Usage: absolute-extension-manifest'; exit 2; fi
EXTENSION_MANIFEST=$1
cd /mnt/beegfs/home/MMReco2021/liuyu/GRID-sensitivity-multiseed
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false USE_TF=0
export GRID_EXPORT_CPU_BUDGET=4 GRID_CPU_BUDGET=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export MPLCONFIGDIR="/tmp/grid-content-verified-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
: "${CONTENT_EXTENSION_MANIFEST_SHA256:?Required extension manifest hash}"
: "${CONTENT_EXTENSION_GIT_COMMIT:?Required extension source commit}"
# This commit describes the unchanged inference source, not the extension's HEAD.
export SOURCE_GIT_COMMIT=1c8fbe94b2d32d5f3d47e9aaf6c4f68140ae9579
export CONTENT_MANIFEST_SHA256=b3bdced637c5ef0744b7452896dade6b2f89a69003c76e75befab2a56c503322
CONTENT_PYTHON=/home/MMReco2021/.conda/envs/GRID/bin/python
srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,cores --kill-on-bad-exit=1 \
    "$CONTENT_PYTHON" -m scripts.retrieval_content_verified run --extension "$EXTENSION_MANIFEST"
# Paths come only from the hash-bound extension, not shell-evaluated file contents.
FULL_OUTPUT_DIR=$("$CONTENT_PYTHON" -c 'import sys; from scripts.retrieval_content_verified import load_extension; print(load_extension(sys.argv[1])["full_output_dir"])' "$EXTENSION_MANIFEST")
PARENT_MANIFEST=/mnt/beegfs/home/MMReco2021/liuyu/GRID/logs/training_launch/retrieval_content_seed42_full_2026-11-07_v1/manifest.json
srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,cores --kill-on-bad-exit=1 \
    "$CONTENT_PYTHON" -m scripts.audit_retrieval_content --manifest "$PARENT_MANIFEST" --output-dir "$FULL_OUTPUT_DIR" --phase full
srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,cores --kill-on-bad-exit=1 \
    "$CONTENT_PYTHON" -m scripts.analyze_retrieval_content --manifest "$PARENT_MANIFEST" --output-dir "$FULL_OUTPUT_DIR" --phase full
srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,cores --kill-on-bad-exit=1 \
    "$CONTENT_PYTHON" -m scripts.retrieval_content_verified accept-full --extension "$EXTENSION_MANIFEST"
