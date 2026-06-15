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

python -m src.train experiment=tiger_train_rerank_small \
    candidate_manifest_path=./outputs/sid_candidates/candidate_manifest.json
