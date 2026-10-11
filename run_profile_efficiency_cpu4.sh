#!/bin/bash
# Fixed same-GPU profiling: MANIFEST OUTPUT_DIR smoke|formal [SMOKE_ACCEPTANCE]
#SBATCH --job-name=GRID-Efficiency-CPU4
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --partition=rtx
#SBATCH --nodelist=aias-compute-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
[[ "$#" -ge 3 && "$#" -le 4 ]] || { echo 'Usage: MANIFEST OUTPUT_DIR smoke|formal [SMOKE_ACCEPTANCE]' >&2; exit 2; }
cd /mnt/beegfs/home/MMReco2021/liuyu/GRID
: "${SOURCE_GIT_COMMIT:?Missing source binding}"
: "${PROFILE_MANIFEST_SHA256:?Missing manifest binding}"
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export GRID_CPU_BUDGET=4 GRID_PROFILE_CPU_BUDGET=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export MPLCONFIGDIR="$(mktemp -d /tmp/grid-efficiency-mpl-XXXXXX)"
PYTHON_BIN=/home/MMReco2021/.conda/envs/GRID/bin/python
if [[ "$3" == smoke ]]; then
    srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,threads --kill-on-bad-exit=1 \
        "$PYTHON_BIN" -m pytest -q tests/test_profile_efficiency_cpu4.py tests/test_export_cpu_budget.py tests/test_repository_launchers.py
fi
extra=()
if [[ "$#" == 4 ]]; then extra=(--smoke-report "$4"); fi
srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,threads --kill-on-bad-exit=1 \
    "$PYTHON_BIN" -m scripts.profile_efficiency_cpu4 --manifest "$1" --output "$2" --phase "$3" "${extra[@]}"
