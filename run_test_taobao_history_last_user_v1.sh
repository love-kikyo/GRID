#!/bin/bash
#SBATCH --job-name=taobao-protocol-test
#SBATCH --output=logs/data_build/taobao_protocol_test_%j.out
#SBATCH --error=logs/data_build/taobao_protocol_test_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:0
#SBATCH --nodelist=aias-compute-2

set -euo pipefail

TASK_ROOT=/mnt/beegfs/home/MMReco2021/liuyu
GRID_ROOT=${TASK_ROOT}/GRID
PYTHON_BIN=/home/MMReco2021/.conda/envs/GRID/bin/python
SMOKE_ROOT=${GRID_ROOT}/data/taobao_history_last_user_v1_smoke_${SLURM_JOB_ID}

cd "${GRID_ROOT}"
mkdir -p logs/data_build

"${PYTHON_BIN}" -m pytest tests/test_build_taobao_next_item_dataset.py -q

"${PYTHON_BIN}" scripts/build_taobao_next_item_dataset.py \
  --split train \
  --data-dir "${TASK_ROOT}/Taobao/taobao" \
  --item2row-path "${TASK_ROOT}/Taobao/cache/sid_item2row_scl_emb_int8_p90.pkl" \
  --output-root "${SMOKE_ROOT}" \
  --min-sequence-length 2 \
  --max-sequence-length 1000 \
  --eval-ratio 0.05 \
  --samples-per-shard 20000 \
  --max-input-shards 1 \
  --max-input-rows 50000

"${PYTHON_BIN}" scripts/build_taobao_next_item_dataset.py \
  --split test \
  --data-dir "${TASK_ROOT}/Taobao/taobao" \
  --item2row-path "${TASK_ROOT}/Taobao/cache/sid_item2row_scl_emb_int8_p90.pkl" \
  --output-root "${SMOKE_ROOT}" \
  --min-sequence-length 2 \
  --max-sequence-length 1000 \
  --samples-per-shard 20000 \
  --max-input-shards 1 \
  --max-input-rows 50000

"${PYTHON_BIN}" scripts/audit_taobao_next_item_dataset.py \
  --data-root "${SMOKE_ROOT}" \
  --expected-protocol history-last-user-v1 \
  --require-partial-run
