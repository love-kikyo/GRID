#!/bin/bash
#
#SBATCH --job-name=GRID-ProfileKV
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=aias-compute-01

set -euo pipefail

PROFILE_JOB_ID="${SLURM_JOB_ID:-local}"
mkdir -p logs/profiles logs/slurm /tmp/matplotlib-${USER}-${PROFILE_JOB_ID}
export MPLCONFIGDIR=/tmp/matplotlib-${USER}-${PROFILE_JOB_ID}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

profile_run() {
    local name="$1"
    local run_dir="$2"
    local variants="$3"

    /home/MMReco2021/.conda/envs/GRID/bin/python scripts/profile_rerank_efficiency.py \
        --run-dir "${run_dir}" \
        --ckpt "${run_dir}/checkpoints/checkpoint_000_085000.ckpt" \
        --output "logs/profiles/${name}_kv_cache_profile.json" \
        --csv-output "logs/profiles/${name}_kv_cache_profile_batches.csv" \
        --variants ${variants} \
        --warmup-batches 5 \
        --measure-batches 30 \
        --batch-size 32 \
        --num-workers 4 \
        --split test \
        --device cuda \
        --amp bf16
}

profile_separate_rerank() {
    local name="$1"
    local sid_run_dir="$2"
    local reranker_ckpt="${3:-}"

    local reranker_args=()
    if [[ -n "${reranker_ckpt}" ]]; then
        reranker_args=(--reranker-ckpt "${reranker_ckpt}")
    fi

    /home/MMReco2021/.conda/envs/GRID/bin/python scripts/profile_rerank_efficiency.py \
        --profile-mode separate_rerank \
        --run-dir "${sid_run_dir}" \
        --ckpt "${sid_run_dir}/checkpoints/checkpoint_000_085000.ckpt" \
        --sid-teacher-ckpt "${sid_run_dir}/checkpoints/checkpoint_000_085000.ckpt" \
        --separate-rerank-config configs/experiment/tiger_train_rerank_small.yaml \
        --output "logs/profiles/${name}_profile.json" \
        --csv-output "logs/profiles/${name}_profile_batches.csv" \
        "${reranker_args[@]}" \
        --variants beam_only separate_rerank \
        --warmup-batches 5 \
        --measure-batches 30 \
        --batch-size 32 \
        --num-workers 4 \
        --split test \
        --device cuda \
        --amp bf16
}

profile_run "listwise" "logs/train/runs/2026-07-11/15-47-00" "beam_only rerank_cache rerank_no_cache"
profile_run "pointwise" "logs/train/runs/2026-06-26/17-00-10" "beam_only rerank_cache rerank_no_cache"
profile_run "sid_beam_only" "logs/train/runs/2026-06-12/16-29-32" "beam_only"
profile_separate_rerank \
    "separate_rerank_small_trained" \
    "logs/train/runs/2026-06-12/16-29-32" \
    "logs/train/runs/2026-07-12/23-28-36/checkpoints/checkpoint_000_020000.ckpt"
