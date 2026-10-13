#!/bin/bash
# Frozen seed42 SIM training acceptance -> export smoke -> full validation.
#SBATCH --job-name=GRID-SIM-Val
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --partition=rtx
#SBATCH --nodelist=aias-compute-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:45:00
set -euo pipefail
[[ "$#" -eq 2 && ( "$1" == smoke || "$1" == full ) && "$2" =~ ^v[0-9]+$ ]] || {
  echo 'Usage: sbatch SCRIPT {smoke|full} vN' >&2; exit 2;
}
MODE=$1
VERSION=$2
GRID_ROOT=/mnt/beegfs/home/MMReco2021/liuyu/GRID
cd "$GRID_ROOT"
PYTHON_BIN=/home/MMReco2021/.conda/envs/GRID/bin/python
export PYTHONPATH="$GRID_ROOT" PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 TOKENIZERS_PARALLELISM=false
export GRID_EXPORT_CPU_BUDGET=4 GRID_CPU_BUDGET=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
: "${SOURCE_GIT_COMMIT:?Missing source binding}"
REPORT_DIR=${GRID_ROOT}/logs/training_launch/sim_seed42_validation_2026-10-13
RUN_DIR=${GRID_ROOT}/logs/train/runs/2026-10-12/sim-soft-train-seed42-139478
CHECKPOINT=${RUN_DIR}/checkpoints/checkpoint_000_010000.ckpt
PREFLIGHT=${GRID_ROOT}/logs/training_preflight/sim_soft_train_seed42_139478.json
CHECKPOINT_ACCEPTANCE=${SIM_CHECKPOINT_ACCEPTANCE:-${REPORT_DIR}/checkpoint_acceptance_${VERSION}.json}
METHOD_TAG=sim_soft_seed42_139478
OUTPUT_TAG=_${MODE}_${VERSION}
OUTPUT=${GRID_ROOT}/logs/analysis/paired_evaluation/${METHOD_TAG}_val${OUTPUT_TAG}.h5
ACCEPTANCE=${REPORT_DIR}/validation_${MODE}_${VERSION}.json
mkdir -p "$REPORT_DIR" logs/slurm "/tmp/matplotlib-${USER}-${SLURM_JOB_ID}"
export MPLCONFIGDIR="/tmp/matplotlib-${USER}-${SLURM_JOB_ID}"
printf 'SIM validation mode=%s source=%s job=%s\n' "$MODE" "$SOURCE_GIT_COMMIT" "$SLURM_JOB_ID"
EXTRA_ACCEPT=()
if [[ "$MODE" == smoke ]]; then
  srun --ntasks=1 --cpus-per-task=1 --cpu-bind=verbose,threads --kill-on-bad-exit=1 \
    "$PYTHON_BIN" -m pytest -q tests/test_sim_export.py tests/test_sim_checkpoint.py tests/test_sim_validation.py \
    tests/test_paired_evaluation.py tests/test_export_cpu_budget.py tests/test_export_user_identity.py tests/test_sim_validation_launcher.py
  srun --ntasks=1 --cpus-per-task=1 --cpu-bind=verbose,threads --kill-on-bad-exit=1 \
    "$PYTHON_BIN" -m scripts.accept_sim_checkpoint \
    --launch logs/training_launch/sim_soft_seed42_139478.json --output "$CHECKPOINT_ACCEPTANCE"
  export MAX_BATCHES=2
  EXTRA_ACCEPT=(--smoke)
else
  : "${SIM_SMOKE_ACCEPTANCE:?Missing accepted smoke}"
  : "${SIM_SMOKE_ACCEPTANCE_SHA256:?Missing smoke hash}"
  # This stage never reruns an existing export or overwrites acceptance files.
  srun --ntasks=1 --cpus-per-task=1 --cpu-bind=verbose,threads --kill-on-bad-exit=1 \
    "$PYTHON_BIN" -c 'import hashlib,json,os,sys; from pathlib import Path; p=Path(os.environ["SIM_SMOKE_ACCEPTANCE"]); a=json.loads(p.read_text()); assert hashlib.sha256(p.read_bytes()).hexdigest()==os.environ["SIM_SMOKE_ACCEPTANCE_SHA256"]; assert a["status"]=="passed" and a["num_samples"]==64 and a["frozen_teacher_beam_exact"]; assert a["checkpoint_acceptance_sha256"]==hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()' "$CHECKPOINT_ACCEPTANCE"
  export MAX_BATCHES=
fi
export OUTPUT_TAG SOURCE_COMPATIBILITY=
bash run_export_paired_evaluation.sh "$METHOD_TAG" val "$RUN_DIR" "$CHECKPOINT" "$PREFLIGHT"
srun --ntasks=1 --cpus-per-task=1 --cpu-bind=verbose,threads --kill-on-bad-exit=1 \
  "$PYTHON_BIN" -m scripts.accept_sim_validation --input "$OUTPUT" \
  --checkpoint-acceptance "$CHECKPOINT_ACCEPTANCE" \
  --reference-selection logs/analysis/paired_evaluation/separate_seed42_132051_fusion_val.json \
  --output "$ACCEPTANCE" "${EXTRA_ACCEPT[@]}"
