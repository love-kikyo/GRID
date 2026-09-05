#!/bin/bash
#SBATCH --job-name=GRID-SID-HLastV1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:3
#SBATCH --nodelist=aias-compute-01

set -euo pipefail

TASK_ROOT=/mnt/beegfs/home/MMReco2021/liuyu
GRID_ROOT=${TASK_ROOT}/GRID
PYTHON_BIN=/home/MMReco2021/.conda/envs/GRID/bin/python
DATA_ROOT=${GRID_ROOT}/data/taobao_history_last_user_v1
SEMANTIC_ID_PATH=${GRID_ROOT}/data/taobao/semantic_id/sid_reindexing_scl_emb_int8_p90.npy
JOB_ID=${SLURM_JOB_ID:-local}
RUN_ID="$(date +%F)/sid-seed42-${JOB_ID}"
PREFLIGHT_REPORT=${GRID_ROOT}/logs/training_preflight/sid_seed42_${JOB_ID}.json

cd "${GRID_ROOT}"
mkdir -p logs/slurm logs/training_preflight "/tmp/matplotlib-${USER}-${JOB_ID}"
export MPLCONFIGDIR="/tmp/matplotlib-${USER}-${JOB_ID}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

"${PYTHON_BIN}" scripts/verify_taobao_training_data.py \
  --data-root "${DATA_ROOT}" \
  --semantic-id-path "${SEMANTIC_ID_PATH}" \
  --num-hierarchies 4 \
  --num-embeddings-per-hierarchy 4096 \
  --source-file scripts/verify_taobao_training_data.py \
  --source-file src/data/loading/components/dataloading.py \
  --source-file src/data/loading/components/interfaces.py \
  --source-file src/models/modules/semantic_id/tiger_generation_model.py \
  --source-file configs/experiment/tiger_train_flat.yaml \
  --source-file configs/experiment/tiger_train_sid.yaml \
  --source-file run_sid_history_last_user_v1.sh \
  --report-path "${PREFLIGHT_REPORT}"

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
  'tags=[taobao-history-last-user-v1,tiger-sid-stage,seed42]' \
  data_loading.train_dataloader_config.dataloader.assign_all_files_per_worker=false \
  data_loading.val_dataloader_config.dataloader.num_workers=2 \
  data_loading.test_dataloader_config.dataloader.num_workers=3 \
  trainer.max_steps=120000 \
  optim.scheduler.num_training_steps=120000
