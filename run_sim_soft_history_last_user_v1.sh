#!/bin/bash
# Explicit mode: smoke (2 steps) or train (120k cap); never changes legacy Separate.
#SBATCH --job-name=GRID-SIM-HLastV1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --partition=rtx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:3
#SBATCH --mem=60G
#SBATCH --time=00:20:00
#SBATCH --nodelist=aias-compute-2
set -euo pipefail
[[ "$#" -ge 2 && "$#" -le 3 ]] || { echo 'Usage: sbatch [--time=100:00:00] SCRIPT {smoke|train} TEACHER [SEED]' >&2; exit 2; }
MODE=$1
TEACHER_CKPT=$2
SEED=${3:-42}
[[ ( "$MODE" == smoke || "$MODE" == train ) && -f "$TEACHER_CKPT" && "$SEED" =~ ^[0-9]+$ ]] || exit 2
GRID_ROOT=/mnt/beegfs/home/MMReco2021/liuyu/GRID
PYTHON_BIN=/home/MMReco2021/.conda/envs/GRID/bin/python
DATA_ROOT=${GRID_ROOT}/data/taobao_history_last_user_v1
SEMANTIC_ID_PATH=${GRID_ROOT}/data/taobao/semantic_id/sid_reindexing_scl_emb_int8_p90.npy
cd "$GRID_ROOT"
: "${SOURCE_GIT_COMMIT:?Submit from a clean, committed source tree}"
JOB_ID=${SLURM_JOB_ID:?Slurm allocation required}
RUN_ID="$(date +%F)/sim-soft-${MODE}-seed${SEED}-${JOB_ID}"
REPORT_DIR=${GRID_ROOT}/logs/stage_smoke/sim_soft_${JOB_ID}
mkdir -p logs/slurm logs/training_preflight "$REPORT_DIR" "/tmp/matplotlib-${USER}-${JOB_ID}"
export MPLCONFIGDIR="/tmp/matplotlib-${USER}-${JOB_ID}"
export PYTHONPATH="$GRID_ROOT" PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 TOKENIZERS_PARALLELISM=false
export USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source scripts/cpu_budget.sh
printf 'SIM source=%s mode=%s seed=%s run=%s\n' "$SOURCE_GIT_COMMIT" "$MODE" "$SEED" "$RUN_ID"
SOURCES=(scripts/verify_taobao_training_data.py scripts/cpu_budget.sh src/runtime_cpu.py
  src/train.py src/checkpoint_resume.py src/data/loading/components/iterators.py
  src/data/loading/components/dataloading.py src/data/loading/components/interfaces.py
  src/models/modules/semantic_id/tiger_generation_model.py
  src/models/modules/semantic_id/separate_rerank_model.py
  src/models/modules/semantic_id/sim_rerank_model.py src/baselines/sim_soft.py
  configs/experiment/tiger_train_flat.yaml configs/experiment/tiger_train_sim_soft.yaml
  scripts/verify_stage_smoke.py scripts/verify_sim_smoke.py
  tests/test_sim_soft.py tests/test_sim_adapter.py tests/test_sim_launcher.py run_sim_soft_history_last_user_v1.sh)
SOURCE_ARGS=()
for file in "${SOURCES[@]}"; do SOURCE_ARGS+=(--source-file "$file"); done
"$PYTHON_BIN" scripts/verify_taobao_training_data.py --data-root "$DATA_ROOT" \
  --semantic-id-path "$SEMANTIC_ID_PATH" --num-hierarchies 4 --num-embeddings-per-hierarchy 4096 \
  --initialization-checkpoint "$TEACHER_CKPT" "${SOURCE_ARGS[@]}" \
  --report-path "logs/training_preflight/sim_soft_${MODE}_seed${SEED}_${JOB_ID}.json"
EXTRA_ARGS=()
if [[ "$MODE" == smoke ]]; then
  srun --ntasks=1 --ntasks-per-node=1 --cpus-per-task=1 --cpu-bind=verbose,threads --kill-on-bad-exit=1 \
    "$PYTHON_BIN" -m pytest -q tests/test_sim_soft.py tests/test_sim_adapter.py tests/test_sim_launcher.py tests/test_stage_smoke.py tests/test_repository_launchers.py
  EXTRA_ARGS=(trainer.max_steps=2 trainer.val_check_interval=4 trainer.limit_val_batches=1
    trainer.log_every_n_steps=1 callbacks.model_checkpoint=null callbacks.early_stopping=null
    '+callbacks.sim_smoke._target_=scripts.verify_sim_smoke.SIMSmokeAudit'
    "+callbacks.sim_smoke.teacher_checkpoint=${TEACHER_CKPT}"
    "+callbacks.sim_smoke.report_dir=${REPORT_DIR}")
fi
srun --ntasks=3 --ntasks-per-node=3 --cpus-per-task=1 --cpu-bind=verbose,threads --kill-on-bad-exit=1 \
  "$PYTHON_BIN" -m src.train 'experiment=[tiger_train_flat,tiger_train_sim_soft]' \
  data_dir="$DATA_ROOT" semantic_id_path="$SEMANTIC_ID_PATH" num_hierarchies=4 num_embeddings_per_hierarchy=4096 \
  seed="$SEED" id="$RUN_ID" ++should_skip_retry=true test=false \
  "tags=[taobao-history-last-user-v1,sid-sim-soft,${MODE},seed${SEED}]" \
  model.sid_teacher_ckpt_path="$TEACHER_CKPT" \
  data_loading.train_dataloader_config.dataloader.assign_all_files_per_worker=false \
  data_loading.train_dataloader_config.dataloader.num_workers=1 \
  data_loading.val_dataloader_config.dataloader.num_workers=2 \
  data_loading.test_dataloader_config.dataloader.num_workers=3 \
  optim.scheduler.num_training_steps=120000 "${EXTRA_ARGS[@]}"
