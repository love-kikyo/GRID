#!/bin/bash
#SBATCH --job-name=GRID-retrain-tests
#SBATCH --output=logs/data_build/retraining_controls_%j.out
#SBATCH --error=logs/data_build/retraining_controls_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:0
#SBATCH --nodelist=aias-compute-2

set -euo pipefail

GRID_ROOT=/mnt/beegfs/home/MMReco2021/liuyu/GRID
PYTHON_BIN=/home/MMReco2021/.conda/envs/GRID/bin/python
SMOKE_ROOT=${GRID_ROOT}/data/taobao_history_last_user_v1_smoke_131855
TEACHER_CKPT=${GRID_ROOT}/logs/train/runs/2026-06-12/16-29-32/checkpoints/checkpoint_000_085000.ckpt

cd "${GRID_ROOT}"
mkdir -p logs/data_build
export PYTHONUNBUFFERED=1

"${PYTHON_BIN}" -m pytest -q \
  tests/test_build_taobao_next_item_dataset.py \
  tests/test_dataloading_empty_worker.py \
  tests/test_history_occurrence_analysis.py \
  tests/test_paired_evaluation.py \
  tests/test_verify_taobao_training_data.py \
  tests/test_rerank_objective.py

"${PYTHON_BIN}" scripts/audit_taobao_next_item_dataset.py \
  --data-root "${SMOKE_ROOT}" \
  --expected-protocol history-last-user-v1 \
  --require-partial-run

"${PYTHON_BIN}" -m src.train \
  "experiment=[tiger_train_flat,tiger_train_click]" \
  data_dir="${SMOKE_ROOT}" \
  semantic_id_path=./data/taobao/semantic_id/sid_reindexing_scl_emb_int8_p90.npy \
  num_hierarchies=4 \
  num_embeddings_per_hierarchy=4096 \
  model.manual_ckpt_path="${TEACHER_CKPT}" \
  --cfg job --resolve

"${PYTHON_BIN}" -m src.train \
  "experiment=[tiger_train_flat,tiger_train_click,tiger_train_click_pointwise]" \
  data_dir="${SMOKE_ROOT}" \
  semantic_id_path=./data/taobao/semantic_id/sid_reindexing_scl_emb_int8_p90.npy \
  num_hierarchies=4 \
  num_embeddings_per_hierarchy=4096 \
  model.manual_ckpt_path="${TEACHER_CKPT}" \
  --cfg job --resolve
