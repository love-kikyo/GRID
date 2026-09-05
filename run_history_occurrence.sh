#!/bin/bash
#
# Usage:
#   sbatch run_history_occurrence.sh dataset RUN_DIR ANALYSIS_RUN_NAME
#   sbatch run_history_occurrence.sh prediction RUN_DIR CHECKPOINT PREFLIGHT ANALYSIS_RUN_NAME
#   sbatch run_history_occurrence.sh all RUN_DIR CHECKPOINT PREFLIGHT ANALYSIS_RUN_NAME

#SBATCH --job-name=GRID-HistOcc
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=aias-compute-01

set -euo pipefail

PHASE="${1:-}"
case "${PHASE}" in
    dataset)
        if [[ "$#" -ne 3 ]]; then
            echo "Usage: sbatch $0 dataset RUN_DIR ANALYSIS_RUN_NAME" >&2
            exit 2
        fi
        ;;
    all|prediction)
        if [[ "$#" -ne 5 ]]; then
            echo "Usage: sbatch $0 ${PHASE} RUN_DIR CHECKPOINT PREFLIGHT ANALYSIS_RUN_NAME" >&2
            exit 2
        fi
        ;;
    *)
        echo "Unknown phase: ${PHASE}. Expected all, dataset, or prediction." >&2
        exit 2
        ;;
esac

GRID_ROOT=/mnt/beegfs/home/MMReco2021/liuyu/GRID
PYTHON_BIN=/home/MMReco2021/.conda/envs/GRID/bin/python
DATA_ROOT=${GRID_ROOT}/data/taobao_history_last_user_v1
RUN_DIR="$2"
if [[ "${PHASE}" == "dataset" ]]; then
    CHECKPOINT=""
    PREFLIGHT=""
    ANALYSIS_RUN_NAME="$3"
else
    CHECKPOINT="$3"
    PREFLIGHT="$4"
    ANALYSIS_RUN_NAME="$5"
fi
for path_argument in "${RUN_DIR}" ${CHECKPOINT:+"${CHECKPOINT}"} ${PREFLIGHT:+"${PREFLIGHT}"}; do
    if [[ "${path_argument}" != /* ]]; then
        echo "Run, checkpoint, and preflight paths must be absolute: ${path_argument}" >&2
        exit 2
    fi
done
if ! [[ "${ANALYSIS_RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ANALYSIS_RUN_NAME contains unsafe characters: ${ANALYSIS_RUN_NAME}" >&2
    exit 2
fi

cd "${GRID_ROOT}"
OUTPUT_ROOT="logs/analysis/history_occurrence/${ANALYSIS_RUN_NAME}"
AUDIT_CACHE="${OUTPUT_ROOT}/semantic_itemkey_collision_audit.json"
JOB_ID="${SLURM_JOB_ID:-local}"

mkdir -p "${OUTPUT_ROOT}" logs/slurm "/tmp/matplotlib-${USER}-${JOB_ID}"
export MPLCONFIGDIR="/tmp/matplotlib-${USER}-${JOB_ID}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

if [[ "${PHASE}" == "all" || "${PHASE}" == "dataset" ]]; then
    "${PYTHON_BIN}" scripts/analyze_history_occurrence.py \
        --mode dataset \
        --run-dir "${RUN_DIR}" \
        --expected-data-root "${DATA_ROOT}" \
        --splits train val test \
        --output-dir "${OUTPUT_ROOT}/dataset" \
        --collision-audit-cache "${AUDIT_CACHE}" \
        --batch-size 512 \
        --num-workers 12 \
        --progress-every 100
fi

if [[ "${PHASE}" == "all" || "${PHASE}" == "prediction" ]]; then
    "${PYTHON_BIN}" scripts/analyze_history_occurrence.py \
        --mode prediction \
        --run-dir "${RUN_DIR}" \
        --checkpoint "${CHECKPOINT}" \
        --training-preflight "${PREFLIGHT}" \
        --expected-data-root "${DATA_ROOT}" \
        --splits test \
        --output-dir "${OUTPUT_ROOT}/prediction_test" \
        --collision-audit-cache "${AUDIT_CACHE}" \
        --batch-size 32 \
        --num-workers 8 \
        --device cuda \
        --amp bf16 \
        --ks 1 5 10 20 \
        --progress-every 25
fi
