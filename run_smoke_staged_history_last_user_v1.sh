#!/bin/bash
# Usage: sbatch run_smoke_staged_history_last_user_v1.sh {listwise|pointwise|separate} TEACHER_CKPT
# Uses the normal src.train entry, formal model/batch settings, two optimizer
# steps and two one-batch validations. No checkpoint or publication result.
#SBATCH --job-name=GRID-Stage-Smoke
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
if [[ "$#" -ne 2 || ! -f "$2" ]]; then
    echo 'Usage: sbatch run_smoke_staged_history_last_user_v1.sh {listwise|pointwise|separate} TEACHER_CKPT' >&2
    exit 2
fi
METHOD=$1
TEACHER_CKPT=$2
GRID_ROOT=/mnt/beegfs/home/MMReco2021/liuyu/GRID
PYTHON_BIN=/home/MMReco2021/.conda/envs/GRID/bin/python
DATA_ROOT=${GRID_ROOT}/data/taobao_history_last_user_v1
SEMANTIC_ID_PATH=${GRID_ROOT}/data/taobao/semantic_id/sid_reindexing_scl_emb_int8_p90.npy
case "${METHOD}" in
    listwise) EXPERIMENT='[tiger_train_flat,tiger_train_click]'; TEACHER_ARG=model.manual_ckpt_path ;;
    pointwise) EXPERIMENT='[tiger_train_flat,tiger_train_click,tiger_train_click_pointwise]'; TEACHER_ARG=model.manual_ckpt_path ;;
    separate) EXPERIMENT='[tiger_train_flat,tiger_train_rerank_small]'; TEACHER_ARG=model.sid_teacher_ckpt_path ;;
    *) echo 'Unknown staged method' >&2; exit 2 ;;
esac
JOB_ID=${SLURM_JOB_ID:-local}
RUN_ID="$(date +%F)/stage-smoke-${METHOD}-${JOB_ID}"
REPORT_DIR=${GRID_ROOT}/logs/stage_smoke/${METHOD}_${JOB_ID}
cd "${GRID_ROOT}"
mkdir -p logs/slurm logs/training_preflight "${REPORT_DIR}" "/tmp/matplotlib-${USER}-${JOB_ID}"
export MPLCONFIGDIR="/tmp/matplotlib-${USER}-${JOB_ID}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
if [[ "${SLURM_NTASKS:-}" != 3 || "${SLURM_NTASKS_PER_NODE:-}" != 3 ]]; then
    echo 'Expected three Slurm tasks for 3-GPU DDP' >&2
    exit 1
fi
"${PYTHON_BIN}" scripts/verify_taobao_training_data.py \
    --data-root "${DATA_ROOT}" --semantic-id-path "${SEMANTIC_ID_PATH}" \
    --num-hierarchies 4 --num-embeddings-per-hierarchy 4096 \
    --initialization-checkpoint "${TEACHER_CKPT}" \
    --source-file scripts/verify_stage_smoke.py \
    --source-file scripts/verify_taobao_training_data.py \
    --source-file src/data/loading/components/dataloading.py \
    --source-file src/data/loading/components/interfaces.py \
    --source-file src/models/modules/semantic_id/tiger_generation_model.py \
    --source-file src/models/modules/semantic_id/separate_rerank_model.py \
    --source-file configs/experiment/tiger_train_flat.yaml \
    --source-file configs/experiment/tiger_train_click.yaml \
    --source-file configs/experiment/tiger_train_click_pointwise.yaml \
    --source-file configs/experiment/tiger_train_rerank_small.yaml \
    --source-file run_smoke_staged_history_last_user_v1.sh \
    --report-path "${GRID_ROOT}/logs/training_preflight/stage_smoke_${METHOD}_${JOB_ID}.json"
srun --ntasks=3 --ntasks-per-node=3 --kill-on-bad-exit=1 \
    "${PYTHON_BIN}" -m src.train "experiment=${EXPERIMENT}" \
    data_dir="${DATA_ROOT}" semantic_id_path="${SEMANTIC_ID_PATH}" \
    num_hierarchies=4 num_embeddings_per_hierarchy=4096 seed=42 id="${RUN_ID}" \
    "${TEACHER_ARG}=${TEACHER_CKPT}" ++should_skip_retry=true \
    "tags=[taobao-history-last-user-v1,stage-smoke,${METHOD}]" \
    data_loading.train_dataloader_config.dataloader.assign_all_files_per_worker=false \
    data_loading.val_dataloader_config.dataloader.num_workers=2 \
    data_loading.test_dataloader_config.dataloader.num_workers=3 \
    trainer.max_steps=2 trainer.val_check_interval=4 trainer.limit_val_batches=1 \
    trainer.num_sanity_val_steps=0 trainer.log_every_n_steps=1 test=false \
    callbacks.model_checkpoint=null callbacks.early_stopping=null \
    '+callbacks.stage_smoke._target_=scripts.verify_stage_smoke.StageSmokeAudit' \
    "+callbacks.stage_smoke.method=${METHOD}" \
    "+callbacks.stage_smoke.teacher_checkpoint=${TEACHER_CKPT}" \
    "+callbacks.stage_smoke.report_dir=${REPORT_DIR}"
