#!/bin/bash
# Full test with already-frozen validation beta; no search or successor jobs.
#SBATCH --job-name=GRID-SIM-FullTest
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --partition=rtx
#SBATCH --nodelist=aias-compute-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
set -euo pipefail
cd /mnt/beegfs/home/MMReco2021/liuyu/GRID
[[ $# == 1 && $1 =~ ^v[0-9]+$ ]] || exit 2
: "${SOURCE_GIT_COMMIT:?Missing source binding}"
[[ $(git rev-parse HEAD) == "$SOURCE_GIT_COMMIT" && -z $(git status --porcelain --untracked-files=normal) ]] || {
  echo 'Source binding requires a clean, matching HEAD at job start' >&2; exit 2;
}
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export GRID_EXPORT_CPU_BUDGET=4 GRID_CPU_BUDGET=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE USE_TF=0
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR=/tmp/matplotlib-${USER}-${SLURM_JOB_ID}
mkdir -p "$MPLCONFIGDIR"
PYTHON_BIN=/home/MMReco2021/.conda/envs/GRID/bin/python
OUTPUT=$PWD/logs/analysis/paired_evaluation/sim_soft_seed42_139478_test_cached_full_$1.h5
REPORT_DIR=$PWD/logs/training_launch/sim_seed42_test_2026-10-14
ACCEPTANCE=$REPORT_DIR/cached_full_$1.json
mkdir -p "$REPORT_DIR"
for target in "$OUTPUT" "${OUTPUT%.h5}.json" "$ACCEPTANCE"; do
  [[ ! -e "$target" ]] || { echo "Refusing existing output: $target" >&2; exit 2; }
done
srun --ntasks=1 --cpus-per-task=1 --cpu-bind=verbose,threads --kill-on-bad-exit=1 \
 "$PYTHON_BIN" -m pytest -q tests/test_sim_cached_test.py tests/test_sim_cached_full.py tests/test_sim_cached_smoke.py tests/test_fusion_cpu_budget.py
srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,cores --kill-on-bad-exit=1 \
 "$PYTHON_BIN" -m scripts.export_sim_cached_test --output "$OUTPUT"
srun --ntasks=1 --cpus-per-task=1 --cpu-bind=verbose,threads --kill-on-bad-exit=1 \
 "$PYTHON_BIN" -m scripts.export_sim_cached_test --output "$OUTPUT" --acceptance "$ACCEPTANCE"
