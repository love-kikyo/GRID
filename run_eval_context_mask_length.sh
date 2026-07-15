#!/bin/bash
#
# Usage:
#   sbatch run_eval_context_mask_length.sh listwise 0
#   sbatch run_eval_context_mask_length.sh listwise 2
#   sbatch run_eval_context_mask_length.sh listwise 5
#   sbatch run_eval_context_mask_length.sh listwise 10
#
# Scheme C: keep top_k_for_score fixed at 10 so the candidate token position
# stays unchanged, and vary active_top_k_for_score by masking lower-ranked
# retrieved history items.

#SBATCH --job-name=GRID-EvalCtxMask
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=aias-compute-01

set -euo pipefail

METHOD="${1:-listwise}"
ACTIVE_TOP_K_SCORE="${2:-10}"
JOB_ID="${SLURM_JOB_ID:-local}"

mkdir -p logs/slurm /tmp/matplotlib-${USER}-${JOB_ID}
export MPLCONFIGDIR=/tmp/matplotlib-${USER}-${JOB_ID}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

case "${METHOD}" in
    listwise)
        EXPERIMENT="[tiger_train_flat,tiger_train_click]"
        CKPT="logs/train/runs/2026-07-11/15-47-00/checkpoints/checkpoint_000_085000.ckpt"
        ;;
    pointwise)
        EXPERIMENT="[tiger_train_flat,tiger_train_click]"
        CKPT="logs/train/runs/2026-06-26/17-00-10/checkpoints/checkpoint_000_085000.ckpt"
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

/home/MMReco2021/.conda/envs/GRID/bin/python -m src.train "experiment=${EXPERIMENT}" \
    train=false \
    test=true \
    ckpt_path="${CKPT}" \
    data_dir=./data/taobao \
    semantic_id_path=./data/taobao/semantic_id/sid_reindexing_scl_emb_int8_p90.npy \
    num_hierarchies=4 \
    num_embeddings_per_hierarchy=4096 \
    model.top_k_for_generation=20 \
    model.top_k_for_score=10 \
    model.active_top_k_for_score="${ACTIVE_TOP_K_SCORE}" \
    eval.evaluators.beam.top_k_list="[5,10,20]" \
    eval.evaluators.rerank.top_k_list="[5,10,20]" \
    id="context_mask_length/${METHOD}_activek${ACTIVE_TOP_K_SCORE}_${JOB_ID}"

