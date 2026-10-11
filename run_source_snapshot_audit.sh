#!/bin/bash
#SBATCH --job-name=GRID-SourceAudit-CPU1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:0
set -euo pipefail
[[ "$#" -eq 2 ]] || { echo 'Usage: MANIFEST OUTPUT_JSON' >&2; exit 2; }
cd /mnt/beegfs/home/MMReco2021/liuyu/GRID
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
: "${SOURCE_GIT_COMMIT:?Missing source binding}"
: "${SOURCE_MANIFEST_SHA256:?Missing manifest binding}"
PYTHON_BIN=/home/MMReco2021/.conda/envs/GRID/bin/python
srun --ntasks=1 --cpus-per-task=1 --cpu-bind=verbose,threads --kill-on-bad-exit=1 \
    "$PYTHON_BIN" -m pytest -q tests/test_source_snapshot_audit.py
srun --ntasks=1 --cpus-per-task=1 --cpu-bind=verbose,threads --kill-on-bad-exit=1 \
    "$PYTHON_BIN" -m scripts.audit_source_snapshot "$1" "$2"
