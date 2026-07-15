#!/bin/bash
#
# Usage:
#   sbatch run_profile_context_mask_length.sh listwise 0
#   sbatch run_profile_context_mask_length.sh listwise 2
#   sbatch run_profile_context_mask_length.sh listwise 5
#   sbatch run_profile_context_mask_length.sh listwise 10
#
# Scheme C: keep top_k_for_score fixed at 10 so the candidate token position
# stays unchanged, and vary active_top_k_for_score by masking lower-ranked
# retrieved history items.

#SBATCH --job-name=GRID-ProfileCtxMask
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
ACTIVE_TOP_K_SCORE="${2:-10}"
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

if (( ACTIVE_TOP_K_SCORE < 0 || ACTIVE_TOP_K_SCORE > 10 )); then
    echo "ACTIVE_TOP_K_SCORE must be in [0, 10], got ${ACTIVE_TOP_K_SCORE}" >&2
    exit 2
fi

/home/MMReco2021/.conda/envs/GRID/bin/python scripts/profile_rerank_efficiency.py \
    --run-dir "${RUN_DIR}" \
    --ckpt "${RUN_DIR}/checkpoints/checkpoint_000_085000.ckpt" \
    --output "logs/profiles/${METHOD}_activek${ACTIVE_TOP_K_SCORE}_context_mask_length_profile.json" \
    --csv-output "logs/profiles/${METHOD}_activek${ACTIVE_TOP_K_SCORE}_context_mask_length_profile_batches.csv" \
    --variants beam_only rerank_cache rerank_no_cache \
    --top-k-for-generation 20 \
    --top-k-for-score 10 \
    --active-top-k-for-score "${ACTIVE_TOP_K_SCORE}" \
    --warmup-batches 5 \
    --measure-batches 30 \
    --batch-size 32 \
    --num-workers 4 \
    --split test \
    --device cuda \
    --amp bf16

