#!/bin/bash
#SBATCH --job-name=GRID-List-HLastV1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:3
#SBATCH --nodelist=aias-compute-01

set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
  echo "Usage: sbatch $0 TEACHER_CKPT [SEED] [RESUME_CKPT]" >&2
  exit 2
fi

TEACHER_CKPT=$1
SEED=${2:-42}
RESUME_CKPT=${3:-}
if [[ -n "${RESUME_CKPT}" && ! -f "${RESUME_CKPT}" ]]; then
  echo "Resume checkpoint not found: ${RESUME_CKPT}" >&2
  exit 2
fi
if [[ "${GRID_RESUME_SMOKE:-0}" != 0 && ( "${GRID_RESUME_SMOKE}" != 1 || -z "${RESUME_CKPT}" ) ]]; then
  echo 'CPU resume smoke requires an explicit resume checkpoint.' >&2
  exit 2
fi
if [ ! -f "${TEACHER_CKPT}" ]; then
  echo "Teacher checkpoint not found: ${TEACHER_CKPT}" >&2
  exit 2
fi
if ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "SEED must be a non-negative integer: ${SEED}" >&2
  exit 2
fi

TASK_ROOT=/mnt/beegfs/home/MMReco2021/liuyu
GRID_ROOT=${TASK_ROOT}/GRID
PYTHON_BIN=/home/MMReco2021/.conda/envs/GRID/bin/python
DATA_ROOT=${GRID_ROOT}/data/taobao_history_last_user_v1
SEMANTIC_ID_PATH=${GRID_ROOT}/data/taobao/semantic_id/sid_reindexing_scl_emb_int8_p90.npy
JOB_ID=${SLURM_JOB_ID:-local}
RUN_ID="$(date +%F)/click-listwise-seed${SEED}-${JOB_ID}"
if [[ "${GRID_RESUME_SMOKE:-0}" == 1 ]]; then
  RUN_ID="$(date +%F)/cpu-resume-smoke-listwise-seed${SEED}-${JOB_ID}"
fi
PREFLIGHT_REPORT=${GRID_ROOT}/logs/training_preflight/listwise_seed${SEED}_${JOB_ID}.json

cd "${GRID_ROOT}"
mkdir -p logs/slurm logs/training_preflight "/tmp/matplotlib-${USER}-${JOB_ID}"
export MPLCONFIGDIR="/tmp/matplotlib-${USER}-${JOB_ID}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
source "${GRID_ROOT}/scripts/cpu_budget.sh"

"${PYTHON_BIN}" scripts/verify_taobao_training_data.py \
  --data-root "${DATA_ROOT}" \
  --semantic-id-path "${SEMANTIC_ID_PATH}" \
  --num-hierarchies 4 \
  --num-embeddings-per-hierarchy 4096 \
  --initialization-checkpoint "${TEACHER_CKPT}" \
  --source-file scripts/verify_taobao_training_data.py \
  --source-file scripts/cpu_budget.sh \
  --source-file src/runtime_cpu.py \
  --source-file src/train.py \
  --source-file src/checkpoint_resume.py \
  --source-file scripts/prepare_training_resume.py \
  --source-file scripts/verify_cpu_resume.py \
  --source-file run_smoke_cpu_resume.sh \
  --source-file src/data/loading/components/iterators.py \
  --source-file src/data/loading/components/dataloading.py \
  --source-file src/data/loading/components/interfaces.py \
  --source-file src/models/modules/semantic_id/tiger_generation_model.py \
  --source-file configs/experiment/tiger_train_flat.yaml \
  --source-file configs/experiment/tiger_train_click.yaml \
  --source-file run_click_listwise_history_last_user_v1.sh \
  --report-path "${PREFLIGHT_REPORT}"

if [[ "${SLURM_NTASKS:-}" != "3" || "${SLURM_NTASKS_PER_NODE:-}" != "3" ]]; then
  echo "Expected a single-node, three-task Slurm allocation for 3-GPU DDP." >&2
  exit 1
fi

MAX_STEPS=120000
RESUME_ARGS=()
if [[ -n "${RESUME_CKPT}" ]]; then
  RUN_DIR=${GRID_ROOT}/logs/train/runs/${RUN_ID}
  RESUME_STEP=$("${PYTHON_BIN}" -m scripts.prepare_training_resume \
    --source "${RESUME_CKPT}" --run-dir "${RUN_DIR}" \
    --teacher "${TEACHER_CKPT}" --seed "${SEED}" --preflight "${PREFLIGHT_REPORT}")
  if ! [[ "${RESUME_STEP}" =~ ^[0-9]+$ ]]; then
    echo 'Resume preparation did not return a valid optimizer step' >&2
    exit 1
  fi
  PREPARED_CKPT=${RUN_DIR}/checkpoints/$(basename "${RESUME_CKPT}")
  SMOKE_STEPS=0
  if [[ "${GRID_RESUME_SMOKE:-0}" == 1 ]]; then
    SMOKE_STEPS=100
    MAX_STEPS=$((RESUME_STEP + SMOKE_STEPS))
    RESUME_ARGS+=(trainer.val_check_interval=400 trainer.limit_val_batches=2 test=false)
  fi
  RESUME_ARGS+=("ckpt_path=${PREPARED_CKPT}" model.load_only_weights=false
    '+callbacks.cpu_resume_audit._target_=scripts.verify_cpu_resume.CPUResumeAudit'
    "+callbacks.cpu_resume_audit.prepared_checkpoint=${PREPARED_CKPT}"
    "+callbacks.cpu_resume_audit.report_dir=${RUN_DIR}/cpu_resume_audit"
    "+callbacks.cpu_resume_audit.expected_step=${RESUME_STEP}"
    "+callbacks.cpu_resume_audit.smoke_steps=${SMOKE_STEPS}")
fi

srun --ntasks=3 --ntasks-per-node=3 --cpus-per-task=1 --cpu-bind=verbose,cores --kill-on-bad-exit=1 \
  "${PYTHON_BIN}" -m src.train "experiment=[tiger_train_flat,tiger_train_click]" \
  data_dir="${DATA_ROOT}" \
  semantic_id_path="${SEMANTIC_ID_PATH}" \
  num_hierarchies=4 \
  num_embeddings_per_hierarchy=4096 \
  seed="${SEED}" \
  id="${RUN_ID}" \
  ++should_skip_retry=true \
  "tags=[taobao-history-last-user-v1,recochain-listwise,seed${SEED}]" \
  model.manual_ckpt_path="${TEACHER_CKPT}" \
  model.rerank_objective=listwise \
  data_loading.train_dataloader_config.dataloader.assign_all_files_per_worker=false \
  data_loading.train_dataloader_config.dataloader.num_workers=1 \
  data_loading.val_dataloader_config.dataloader.num_workers=2 \
  data_loading.test_dataloader_config.dataloader.num_workers=3 \
  trainer.max_steps="${MAX_STEPS}" \
  optim.scheduler.num_training_steps=120000 \
  "${RESUME_ARGS[@]}"
