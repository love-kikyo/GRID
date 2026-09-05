#!/bin/bash
#SBATCH --job-name=GRID-loader-v1
#SBATCH --output=logs/data_build/grid_loader_v1_%j.out
#SBATCH --error=logs/data_build/grid_loader_v1_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:0
#SBATCH --nodelist=aias-compute-2

set -euo pipefail

GRID_ROOT=/mnt/beegfs/home/MMReco2021/liuyu/GRID
PYTHON_BIN=/home/MMReco2021/.conda/envs/GRID/bin/python

cd "${GRID_ROOT}"
mkdir -p logs/data_build
export PYTHONUNBUFFERED=1

"${PYTHON_BIN}" scripts/smoke_taobao_grid_dataloader.py \
  --data-root "${GRID_ROOT}/data/taobao_history_last_user_v1" \
  --semantic-id-path "${GRID_ROOT}/data/taobao/semantic_id/sid_reindexing_scl_emb_int8_p90.npy" \
  --batch-size 4 \
  --num-workers 8
