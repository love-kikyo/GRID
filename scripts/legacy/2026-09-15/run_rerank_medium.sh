#!/bin/bash
# Historical launcher: archived 2026-09-15; see this directory's README.md.
#
#SBATCH --job-name=GRID-RerankMedium
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=80G
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:3
#SBATCH --nodelist=aias-compute-01

set -euo pipefail

JOB_ID="${SLURM_JOB_ID:-local}"
mkdir -p logs/slurm /tmp/matplotlib-${USER}-${JOB_ID}
export MPLCONFIGDIR=/tmp/matplotlib-${USER}-${JOB_ID}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

/home/MMReco2021/.conda/envs/GRID/bin/python -m src.train "experiment=[tiger_train_flat,tiger_train_rerank_medium]" \
    data_dir=./data/taobao \
    semantic_id_path=./data/taobao/semantic_id/sid_reindexing_scl_emb_int8_p90.npy \
    num_hierarchies=4 \
    num_embeddings_per_hierarchy=4096 \
    model.sid_teacher_ckpt_path=/home/MMReco2021/liuyu/GRID/logs/train/runs/2026-06-12/16-29-32/checkpoints/checkpoint_000_085000.ckpt
