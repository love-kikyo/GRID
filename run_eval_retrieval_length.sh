#!/bin/bash
#
# Usage:
#   sbatch run_eval_retrieval_length.sh listwise 5
#   sbatch run_eval_retrieval_length.sh listwise 10
#   sbatch run_eval_retrieval_length.sh listwise 20
#   sbatch run_eval_retrieval_length.sh sid 20
#
# Runs Lightning test-only evaluation while overriding top_k_for_generation.

#SBATCH --job-name=GRID-EvalLen
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
TOP_K="${2:-20}"
JOB_ID="${SLURM_JOB_ID:-local}"

mkdir -p logs/slurm /tmp/matplotlib-${USER}-${JOB_ID}
export MPLCONFIGDIR=/tmp/matplotlib-${USER}-${JOB_ID}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

case "${METHOD}" in
    sid)
        EXPERIMENT="tiger_train_flat"
        CKPT="logs/train/runs/2026-06-12/16-29-32/checkpoints/checkpoint_000_085000.ckpt"
        ;;
    listwise)
        EXPERIMENT="[tiger_train_flat,tiger_train_click]"
        CKPT="logs/train/runs/2026-07-11/15-47-00/checkpoints/checkpoint_000_085000.ckpt"
        ;;
    pointwise)
        EXPERIMENT="[tiger_train_flat,tiger_train_click]"
        CKPT="logs/train/runs/2026-06-26/17-00-10/checkpoints/checkpoint_000_085000.ckpt"
        ;;
    *)
        echo "Unknown method: ${METHOD}. Expected one of: sid, listwise, pointwise" >&2
        exit 2
        ;;
esac

case "${TOP_K}" in
    5)
        TOP_K_LIST="[5]"
        ;;
    10)
        TOP_K_LIST="[5,10]"
        ;;
    20)
        TOP_K_LIST="[5,10,20]"
        ;;
    *)
        echo "Unsupported TOP_K=${TOP_K}. Expected one of: 5, 10, 20" >&2
        exit 2
        ;;
esac

/home/MMReco2021/.conda/envs/GRID/bin/python -m src.train "experiment=${EXPERIMENT}" \
    train=false \
    test=true \
    ckpt_path="${CKPT}" \
    data_dir=./data/taobao \
    semantic_id_path=./data/taobao/semantic_id/sid_reindexing_scl_emb_int8_p90.npy \
    num_hierarchies=4 \
    num_embeddings_per_hierarchy=4096 \
    model.top_k_for_generation="${TOP_K}" \
    eval.evaluators.beam.top_k_list="${TOP_K_LIST}" \
    eval.evaluators.rerank.top_k_list="${TOP_K_LIST}" \
    id="retrieval_length/${METHOD}_k${TOP_K}_${JOB_ID}"

