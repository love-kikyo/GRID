#!/bin/bash
# Submit with GRID_RANDOM_INIT_COMMIT set to the committed execution version.
# One allocation: accepted two-update smoke, then a NEW random-initialized fit.
# No teacher, resume, automatic test, extra seed, or changed model mathematics.
#SBATCH --job-name=GRID-List-Random42
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --partition=rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:4090:3

set -euo pipefail
if [[ "$#" -gt 1 || "${1:-42}" != 42 ]]; then
  echo 'Usage: sbatch --export=ALL,GRID_RANDOM_INIT_COMMIT=<commit> run_click_listwise_random_init.sh [42]' >&2
  exit 2
fi
if ! [[ "${GRID_RANDOM_INIT_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]]; then
  echo 'An explicit committed execution version is required.' >&2
  exit 2
fi
GRID_ROOT=/mnt/beegfs/home/MMReco2021/liuyu/GRID
PYTHON_BIN=/home/MMReco2021/.conda/envs/GRID/bin/python
DATA_ROOT=${GRID_ROOT}/data/taobao_history_last_user_v1
SEMANTIC_ID_PATH=${GRID_ROOT}/data/taobao/semantic_id/sid_reindexing_scl_emb_int8_p90.npy
JOB_ID=${SLURM_JOB_ID:?Submit through Slurm}
RUN_DATE=$(date +%F)
SMOKE_ID=${RUN_DATE}/random-init-smoke-listwise-seed42-${JOB_ID}
FORMAL_ID=${RUN_DATE}/click-listwise-random-seed42-${JOB_ID}
REPORT_ROOT=${GRID_ROOT}/logs/random_init_audit/seed42_${JOB_ID}
PREFLIGHT=${GRID_ROOT}/logs/training_preflight/listwise_random_seed42_${JOB_ID}.json
cd "${GRID_ROOT}"

# Documentation-only commits are allowed after submission; execution drift is not.
TEST_FILES=(tests/test_random_init_audit.py tests/test_random_init_launcher.py tests/test_stage_smoke.py tests/test_runtime_cpu.py tests/test_checkpoint_resume.py)
BOUND_PATHS=(src configs scripts run_click_listwise_random_init.sh "${TEST_FILES[@]}")
git diff --exit-code "${GRID_RANDOM_INIT_COMMIT}" -- "${BOUND_PATHS[@]}"
if [[ -n "$(git status --porcelain -- "${BOUND_PATHS[@]}")" ]]; then
  echo 'Execution sources must be clean and match the submitted version.' >&2
  exit 1
fi
for target in "logs/train/runs/${SMOKE_ID}" "logs/train/runs/${FORMAL_ID}" "${REPORT_ROOT}" "${PREFLIGHT}"; do
  if [[ -e "${target}" ]]; then
    echo "Refusing an existing run or audit path: ${target}" >&2
    exit 1
  fi
done
mkdir -p logs/slurm logs/training_preflight "/tmp/matplotlib-${USER}-${JOB_ID}"
export MPLCONFIGDIR="/tmp/matplotlib-${USER}-${JOB_ID}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
source "${GRID_ROOT}/scripts/cpu_budget.sh"
# Real tensor regression tests run on the allocated compute node, CPU only.
"${PYTHON_BIN}" -m pytest -q "${TEST_FILES[@]}"

# Fingerprint the same formal assets; deliberately do NOT supply a checkpoint.
SOURCE_ARGS=()
while IFS= read -r source_file; do
  SOURCE_ARGS+=(--source-file "${source_file}")
done < <(git ls-files "${BOUND_PATHS[@]}")
"${PYTHON_BIN}" scripts/verify_taobao_training_data.py \
  --data-root "${DATA_ROOT}" --semantic-id-path "${SEMANTIC_ID_PATH}" \
  --num-hierarchies 4 --num-embeddings-per-hierarchy 4096 \
  "${SOURCE_ARGS[@]}" --report-path "${PREFLIGHT}"

run_stage() {
  local stage=$1 run_id=$2
  local stage_args=()
  if [[ "${stage}" == smoke ]]; then
    stage_args+=(trainer.max_steps=2 trainer.val_check_interval=4 trainer.limit_val_batches=1
      trainer.log_every_n_steps=1 callbacks.model_checkpoint=null callbacks.early_stopping=null
      +callbacks.random_init_audit.smoke=true)
  else
    stage_args+=(trainer.max_steps=120000 +callbacks.random_init_audit.smoke=false)
  fi
  srun --ntasks=3 --ntasks-per-node=3 --cpus-per-task=1 --cpu-bind=verbose,cores --kill-on-bad-exit=1 \
    "${PYTHON_BIN}" -m src.train 'experiment=[tiger_train_flat,tiger_train_click]' \
    data_dir="${DATA_ROOT}" semantic_id_path="${SEMANTIC_ID_PATH}" \
    num_hierarchies=4 num_embeddings_per_hierarchy=4096 seed=42 id="${run_id}" \
    'tags=[taobao-history-last-user-v1,recochain-listwise,random-init,seed42]' \
    ckpt_path=null model.load_only_weights=false model.manual_ckpt_path=null \
    model.rerank_objective=listwise callbacks.restart_job=null ++should_skip_retry=true \
    data_loading.train_dataloader_config.dataloader.assign_all_files_per_worker=false \
    data_loading.train_dataloader_config.dataloader.num_workers=1 \
    data_loading.val_dataloader_config.dataloader.num_workers=2 \
    data_loading.test_dataloader_config.dataloader.num_workers=3 \
    optim.scheduler.num_training_steps=120000 test=false \
    '+callbacks.random_init_audit._target_=scripts.verify_random_init.RandomInitAudit' \
    "+callbacks.random_init_audit.report_dir=${REPORT_ROOT}/${stage}" \
    "${stage_args[@]}"
}

run_stage smoke "${SMOKE_ID}"
"${PYTHON_BIN}" -m scripts.verify_random_init --report-dir "${REPORT_ROOT}/smoke" --smoke
# Separate process, fresh RNG seed, fresh optimizer; never resume smoke weights.
run_stage formal "${FORMAL_ID}"
"${PYTHON_BIN}" -m scripts.verify_random_init --report-dir "${REPORT_ROOT}/formal"
