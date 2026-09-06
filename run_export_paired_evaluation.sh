#!/bin/bash
#
# Usage:
#   sbatch run_export_paired_evaluation.sh \
#     METHOD_TAG SPLIT RUN_DIR CHECKPOINT TRAINING_PREFLIGHT
# Optional environment variables:
#   SOURCE_COMPATIBILITY=/abs/reviewed_source_transition.json (default: strict matching)
#   MAX_BATCHES=2 OUTPUT_TAG=_smoke sbatch run_export_paired_evaluation.sh \
#     listwise_seed42 test /abs/run /abs/model.ckpt /abs/preflight.json

#SBATCH --job-name=GRID-PairedExport
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# mmap checkpoint loading keeps listwise host RSS low; the separate reranker
# smoke test peaked below 16 GiB, so 20 GiB allows three jobs under the user QoS.
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=aias-compute-01

set -euo pipefail

if [[ "$#" -ne 5 ]]; then
    echo "Usage: sbatch $0 METHOD_TAG {val|test} RUN_DIR CHECKPOINT TRAINING_PREFLIGHT" >&2
    exit 2
fi

METHOD_TAG="$1"
SPLIT="$2"
RUN_DIR="$3"
CHECKPOINT="$4"
TRAINING_PREFLIGHT="$5"
OUTPUT_TAG="${OUTPUT_TAG:-}"
MAX_BATCHES="${MAX_BATCHES:-}"

GRID_ROOT=/mnt/beegfs/home/MMReco2021/liuyu/GRID
PYTHON_BIN=/home/MMReco2021/.conda/envs/GRID/bin/python

for path_argument in "${RUN_DIR}" "${CHECKPOINT}" "${TRAINING_PREFLIGHT}"; do
    if [[ "${path_argument}" != /* ]]; then
        echo "Run directory, checkpoint, and preflight paths must be absolute: ${path_argument}" >&2
        exit 2
    fi
done

cd "${GRID_ROOT}"

if ! [[ "${METHOD_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "METHOD_TAG may contain only letters, digits, dot, underscore, and hyphen." >&2
    exit 2
fi

if [[ "${SPLIT}" != "val" && "${SPLIT}" != "test" ]]; then
    echo "Unknown split: ${SPLIT}. Expected val or test." >&2
    exit 2
fi
if [[ ! -d "${RUN_DIR}" || ! -f "${RUN_DIR}/.hydra/config.yaml" ]]; then
    echo "Invalid run directory: ${RUN_DIR}" >&2
    exit 2
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT}" >&2
    exit 2
fi
if [[ ! -f "${TRAINING_PREFLIGHT}" ]]; then
    echo "Training preflight not found: ${TRAINING_PREFLIGHT}" >&2
    exit 2
fi

JOB_ID="${SLURM_JOB_ID:-local}"
OUTPUT_DIR="logs/analysis/paired_evaluation"
mkdir -p logs/slurm "${OUTPUT_DIR}" "/tmp/matplotlib-${USER}-${JOB_ID}"
export MPLCONFIGDIR="/tmp/matplotlib-${USER}-${JOB_ID}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
if [[ -z "${SOURCE_GIT_COMMIT:-}" ]]; then
    SOURCE_GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"
fi
export SOURCE_GIT_COMMIT

EXTRA_ARGS=()
if [[ -n "${MAX_BATCHES}" ]]; then
    EXTRA_ARGS+=(--max-batches "${MAX_BATCHES}")
fi
if [[ -n "${SOURCE_COMPATIBILITY:-}" ]]; then
    EXTRA_ARGS+=(--source-compatibility "${SOURCE_COMPATIBILITY}")
fi

"${PYTHON_BIN}" scripts/export_paired_evaluation.py \
    --run-dir "${RUN_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --training-preflight "${TRAINING_PREFLIGHT}" \
    --expected-data-root "/mnt/beegfs/home/MMReco2021/liuyu/GRID/data/taobao_history_last_user_v1" \
    --expected-protocol history-last-user-v1 \
    --split "${SPLIT}" \
    --output "${OUTPUT_DIR}/${METHOD_TAG}_${SPLIT}${OUTPUT_TAG}.h5" \
    --batch-size 32 \
    --num-workers 1 \
    --flush-samples 8192 \
    --device cuda \
    --amp bf16 \
    "${EXTRA_ARGS[@]}"
