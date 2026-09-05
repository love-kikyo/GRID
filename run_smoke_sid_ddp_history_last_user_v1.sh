#!/bin/bash
#SBATCH --job-name=GRID-DDP3-Smoke
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:3
#SBATCH --nodelist=aias-compute-01

set -euo pipefail

TASK_ROOT=/mnt/beegfs/home/MMReco2021/liuyu
GRID_ROOT=${TASK_ROOT}/GRID
PYTHON_BIN=/home/MMReco2021/.conda/envs/GRID/bin/python
DATA_ROOT=${GRID_ROOT}/data/taobao_history_last_user_v1
SEMANTIC_ID_PATH=${GRID_ROOT}/data/taobao/semantic_id/sid_reindexing_scl_emb_int8_p90.npy
JOB_ID=${SLURM_JOB_ID:-local}
RUN_ID="$(date +%F)/ddp3-smoke-${JOB_ID}"

cd "${GRID_ROOT}"
mkdir -p logs/slurm "/tmp/matplotlib-${USER}-${JOB_ID}"
export MPLCONFIGDIR="/tmp/matplotlib-${USER}-${JOB_ID}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

if [[ "${SLURM_NTASKS:-}" != "3" || "${SLURM_NTASKS_PER_NODE:-}" != "3" ]]; then
  echo "Expected a single-node, three-task Slurm allocation for 3-GPU DDP." >&2
  exit 1
fi

srun --ntasks=3 --ntasks-per-node=3 --kill-on-bad-exit=1 \
  "${PYTHON_BIN}" -m src.train "experiment=[tiger_train_flat,tiger_train_sid]" \
  data_dir="${DATA_ROOT}" \
  semantic_id_path="${SEMANTIC_ID_PATH}" \
  num_hierarchies=4 \
  num_embeddings_per_hierarchy=4096 \
  seed=42 \
  id="${RUN_ID}" \
  ++should_skip_retry=true \
  'tags=[taobao-history-last-user-v1,ddp3-smoke]' \
  data_loading.train_dataloader_config.dataloader.assign_all_files_per_worker=false \
  data_loading.train_dataloader_config.dataloader.num_workers=8 \
  data_loading.val_dataloader_config.dataloader.num_workers=2 \
  data_loading.test_dataloader_config.dataloader.num_workers=3 \
  trainer.max_steps=2 \
  trainer.num_sanity_val_steps=0 \
  trainer.limit_val_batches=0 \
  trainer.log_every_n_steps=1 \
  optim.scheduler.num_training_steps=2 \
  test=false
