#!/bin/bash
# Usage: sbatch run_select_score_fusion.sh METHOD_TAG VALIDATION_EXPORT_H5

#SBATCH --job-name=GRID-FusionSelect
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:0

set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "Usage: sbatch $0 METHOD_TAG VALIDATION_EXPORT_H5" >&2
    exit 2
fi

METHOD_TAG="$1"
VALIDATION_EXPORT="$2"
if [[ "${VALIDATION_EXPORT}" != /* ]]; then
    echo "Validation export path must be absolute: ${VALIDATION_EXPORT}" >&2
    exit 2
fi
if ! [[ "${METHOD_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "METHOD_TAG may contain only letters, digits, dot, underscore, and hyphen." >&2
    exit 2
fi
if [[ ! -f "${VALIDATION_EXPORT}" || ! -f "${VALIDATION_EXPORT%.h5}.json" ]]; then
    echo "Validation export or sidecar is missing: ${VALIDATION_EXPORT}" >&2
    exit 2
fi

GRID_ROOT=/mnt/beegfs/home/MMReco2021/liuyu/GRID
PYTHON_BIN=/home/MMReco2021/.conda/envs/GRID/bin/python
ANALYSIS_DIR=${GRID_ROOT}/logs/analysis/paired_evaluation

cd "${GRID_ROOT}"
mkdir -p logs/slurm "${ANALYSIS_DIR}"
export PYTHONUNBUFFERED=1
export GRID_CPU_BUDGET=4 GRID_FUSION_CPU_BUDGET=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE

srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,cores --kill-on-bad-exit=1 \
    "${PYTHON_BIN}" -m scripts.select_score_fusion \
    --input "${VALIDATION_EXPORT}" \
    --beam-weight 1.0 \
    --rerank-weight-start 0.00 \
    --rerank-weight-stop 0.50 \
    --rerank-weight-step 0.01 \
    --metric-order ndcg@10 ndcg@5 recall@10 recall@5 \
    --sample-modulus 1 \
    --sample-remainder 0 \
    --output-json "${ANALYSIS_DIR}/${METHOD_TAG}_fusion_val.json" \
    --output-md "${ANALYSIS_DIR}/${METHOD_TAG}_fusion_val.md"
