#!/bin/bash
# Re-audit the immutable 144414 report, fixing only metric display order.
#SBATCH --job-name=GRID-SIM-PairAudit
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --partition=rtx
#SBATCH --nodelist=aias-compute-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
cd /mnt/beegfs/home/MMReco2021/liuyu/GRID
: "${SOURCE_GIT_COMMIT:?Missing source binding}"
: "${SIM_TEST_ACCEPTANCE_SHA256:?Missing accepted test binding}"
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1 USE_TF=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
PYTHON_BIN=/home/MMReco2021/.conda/envs/GRID/bin/python
srun --ntasks=1 --cpus-per-task=1 --cpu-bind=verbose,threads --kill-on-bad-exit=1 \
 "$PYTHON_BIN" -m pytest -q tests/test_sim_paired.py tests/test_paired_cpu_budget.py tests/test_paired_evaluation.py
srun --ntasks=1 --cpus-per-task=1 --cpu-bind=verbose,threads --kill-on-bad-exit=1 \
 "$PYTHON_BIN" -m scripts.analyze_sim_seed42_paired audit-recovery
