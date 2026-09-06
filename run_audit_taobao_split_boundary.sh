#!/bin/bash
# CPU-only first-time cross-split audit. Does not rebuild data or retrain models.
#SBATCH --job-name=GRID-SplitBoundary
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:0
#SBATCH --nodelist=aias-compute-2

set -euo pipefail
GRID_ROOT=/mnt/beegfs/home/MMReco2021/liuyu/GRID
PYTHON_BIN=/home/MMReco2021/.conda/envs/GRID/bin/python
RUN_DIR=${GRID_ROOT}/logs/train/runs/2026-09-05/sid-seed42-131908
PREFLIGHT=${GRID_ROOT}/logs/training_preflight/sid_seed42_131908.json
JOB_ID=${SLURM_JOB_ID:-local}
OUTPUT_DIR=${GRID_ROOT}/logs/analysis/split_boundary
cd "${GRID_ROOT}"
mkdir -p logs/slurm "${OUTPUT_DIR}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=''
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=2
export OMP_NUM_THREADS=4

"${PYTHON_BIN}" -m pytest -q tests/test_audit_taobao_split_boundary.py
COMMON_ARGS=(--data-root "${GRID_ROOT}/data/taobao_history_last_user_v1"
             --run-dir "${RUN_DIR}" --training-preflight "${PREFLIGHT}" --batch-size 1024)
"${PYTHON_BIN}" scripts/audit_taobao_split_boundary.py "${COMMON_ARGS[@]}" \
    --max-records-per-split 128 --output "${OUTPUT_DIR}/boundary_${JOB_ID}_smoke.json"
"${PYTHON_BIN}" scripts/audit_taobao_split_boundary.py "${COMMON_ARGS[@]}" \
    --output "${OUTPUT_DIR}/boundary_${JOB_ID}.json"
