#!/bin/bash
#
#SBATCH --job-name=GRID-SID-CAND
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=60G
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:3
#SBATCH --nodelist=aias-compute-01

python -m src.inference experiment=tiger_infer_sid_candidates \
    data_dir=./data/taobao \
    semantic_id_path=./data/taobao/semantic_id/sid_reindexing_scl_emb_int8_p90.npy \
    ckpt_path=./path/to/trained_sid.ckpt \
    predict_split=testing \
    num_hierarchies=4 \
    num_embeddings_per_hierarchy=4096
