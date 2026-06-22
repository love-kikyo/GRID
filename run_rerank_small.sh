#!/bin/bash
#
#SBATCH --job-name=GRID-RerankSmall
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=60G
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:3
#SBATCH --nodelist=aias-compute-01

python -m src.train "experiment=[tiger_train_flat,tiger_train_rerank_small]" \
    data_dir=./data/taobao \
    semantic_id_path=./data/taobao/semantic_id/sid_reindexing_scl_emb_int8_p90.npy \
    num_hierarchies=4 \
    num_embeddings_per_hierarchy=4096 \
    model.sid_teacher_ckpt_path=/home/MMReco2021/liuyu/GRID/logs/train/runs/2026-06-12/16-29-32/checkpoints/checkpoint_000_085000.ckpt
