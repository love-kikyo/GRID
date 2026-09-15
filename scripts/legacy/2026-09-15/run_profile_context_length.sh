#!/bin/bash
# Historical launcher: archived 2026-09-15; see this directory's README.md.
#
# Usage:
#   sbatch run_profile_context_length.sh listwise 0
#   sbatch run_profile_context_length.sh listwise 5
#   sbatch run_profile_context_length.sh listwise 10
#   sbatch run_profile_context_length.sh listwise 20
#
# Measures model-only latency while overriding top_k_for_score.
# This measures the candidate-aware historical context retrieval length
# used by reranking, not the beam/candidate list length.

#SBATCH --job-name=GRID-ProfileCtx
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=aias-compute-01

set -euo pipefail

METHOD="${1:-listwise}"
TOP_K_SCORE="${2:-10}"
JOB_ID="${SLURM_JOB_ID:-local}"

mkdir -p logs/profiles logs/slurm /tmp/matplotlib-${USER}-${JOB_ID}
export MPLCONFIGDIR=/tmp/matplotlib-${USER}-${JOB_ID}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

case "${METHOD}" in
    listwise)
        RUN_DIR="logs/train/runs/2026-07-11/15-47-00"
        ;;
    pointwise)
        RUN_DIR="logs/train/runs/2026-06-26/17-00-10"
        ;;
    *)
        echo "Unknown method: ${METHOD}. Expected one of: listwise, pointwise" >&2
        exit 2
        ;;
esac

case "${TOP_K_SCORE}" in
    0|5|10|20)
        ;;
    *)
        echo "Unsupported TOP_K_SCORE=${TOP_K_SCORE}. Expected one of: 0, 5, 10, 20" >&2
        exit 2
        ;;
esac

/home/MMReco2021/.conda/envs/GRID/bin/python scripts/profile_rerank_efficiency.py \
    --run-dir "${RUN_DIR}" \
    --ckpt "${RUN_DIR}/checkpoints/checkpoint_000_085000.ckpt" \
    --output "logs/profiles/${METHOD}_scorek${TOP_K_SCORE}_context_length_profile.json" \
    --csv-output "logs/profiles/${METHOD}_scorek${TOP_K_SCORE}_context_length_profile_batches.csv" \
    --variants beam_only rerank_cache rerank_no_cache \
    --top-k-for-generation 20 \
    --top-k-for-score "${TOP_K_SCORE}" \
    --warmup-batches 5 \
    --measure-batches 30 \
    --batch-size 32 \
    --num-workers 4 \
    --split test \
    --device cuda \
    --amp bf16
