#!/bin/bash
#SBATCH --job-name=GRID-BeamCohort-CPU4
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:0
set -euo pipefail
[[ "$#" -eq 2 ]] || { echo 'Usage: MANIFEST OUTPUT_JSON' >&2; exit 2; }
cd /mnt/beegfs/home/MMReco2021/liuyu/GRID
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 GRID_CPU_BUDGET=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
: "${SOURCE_GIT_COMMIT:?Missing source binding}"
: "${COHORT_MANIFEST_SHA256:?Missing manifest binding}"
PYTHON_BIN=/home/MMReco2021/.conda/envs/GRID/bin/python
srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,threads --kill-on-bad-exit=1 \
    "$PYTHON_BIN" -m pytest -q tests/test_beam_cohorts_cpu4.py tests/test_paired_cpu_budget.py
srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,threads --kill-on-bad-exit=1 \
    "$PYTHON_BIN" -m scripts.analyze_beam_cohorts_cpu4 "$1" "$2"
