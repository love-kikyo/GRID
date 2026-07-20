#!/bin/bash
#
#SBATCH --job-name=GRID-JointPairwiseHybrid
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=60G
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:3
#SBATCH --nodelist=aias-compute-01

/home/MMReco2021/.conda/envs/GRID/bin/python -m src.train "experiment=[tiger_train_flat,tiger_train_click,tiger_train_pairwise_hybrid]" \
    data_dir=./data/taobao \
    semantic_id_path=./data/taobao/semantic_id/sid_reindexing_scl_emb_int8_p90.npy \
    num_hierarchies=4 \
    num_embeddings_per_hierarchy=4096
