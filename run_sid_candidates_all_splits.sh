#!/bin/bash
#
#SBATCH --job-name=GRID-SID-CAND-ALL
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=60G
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:3
#SBATCH --nodelist=aias-compute-01

set -euo pipefail

SID_CKPT=/home/MMReco2021/liuyu/GRID/logs/train/runs/2026-06-12/16-29-32/checkpoints/checkpoint_000_085000.ckpt
DATA_DIR=./data/taobao
SEMANTIC_ID_PATH=./data/taobao/semantic_id/sid_reindexing_scl_emb_int8_p90.npy
NUM_HIERARCHIES=4
NUM_EMBEDDINGS_PER_HIERARCHY=4096
TOP_K_FOR_GENERATION=20
ROOT_OUTPUT_DIR=./outputs/sid_candidates
MANIFEST_PATH=${ROOT_OUTPUT_DIR}/candidate_manifest.json

for SPLIT in training evaluation testing; do
  python -m src.inference experiment=tiger_infer_sid_candidates \
      data_dir=${DATA_DIR} \
      semantic_id_path=${SEMANTIC_ID_PATH} \
      ckpt_path=${SID_CKPT} \
      predict_split=${SPLIT} \
      candidate_output_subdir=${SPLIT}_candidate_cache \
      num_hierarchies=${NUM_HIERARCHIES} \
      num_embeddings_per_hierarchy=${NUM_EMBEDDINGS_PER_HIERARCHY} \
      model.top_k_for_generation=${TOP_K_FOR_GENERATION} \
      paths.output_dir=${ROOT_OUTPUT_DIR}
done

python3 -c "from src.utils.candidate_manifest import build_candidate_manifest; build_candidate_manifest(output_path='${MANIFEST_PATH}', root_output_dir='${ROOT_OUTPUT_DIR}', num_hierarchies=${NUM_HIERARCHIES}, num_embeddings_per_hierarchy=${NUM_EMBEDDINGS_PER_HIERARCHY}, top_k_for_generation=${TOP_K_FOR_GENERATION})"
