#!/bin/bash
# Frozen seed42 inference only. Usage: sbatch SCRIPT MANIFEST OUT PHASE GROUP [SMOKE_DIR]
#SBATCH --job-name=GRID-Sensitivity42
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:4090:1
#SBATCH --partition=rtx4090
#SBATCH --nodelist=aias-compute-2
#SBATCH --time=12:00:00
set -euo pipefail
if [[ $# -ne 5 ]]; then
    echo 'Usage: MANIFEST OUTPUT_DIR {smoke|full} {retrieval|beam} SMOKE_DIR_OR_DASH'
    exit 2
fi
MANIFEST=$1
OUTPUT_DIR=$2
PHASE=$3
GROUP=$4
SMOKE_DIR=$5
if [[ "$MANIFEST" != /* || "$OUTPUT_DIR" != /* ]]; then
    echo 'Manifest and output paths must be absolute'; exit 2
fi
if [[ "$PHASE" != smoke && "$PHASE" != full ]] || [[ "$GROUP" != retrieval && "$GROUP" != beam ]]; then
    echo 'Unsupported phase or group'; exit 2
fi
if [[ "$PHASE" == full && "$SMOKE_DIR" != /* ]]; then
    echo 'Full stage requires absolute accepted smoke directory'; exit 2
fi
cd /mnt/beegfs/home/MMReco2021/liuyu/GRID
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false USE_TF=0
export GRID_EXPORT_CPU_BUDGET=4 GRID_CPU_BUDGET=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export MPLCONFIGDIR="/tmp/grid-sensitivity-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
: "${SOURCE_GIT_COMMIT:?Required execution commit}"
: "${SENSITIVITY_MANIFEST_SHA256:?Required frozen manifest hash}"
SENSITIVITY_PYTHON=/home/MMReco2021/.conda/envs/GRID/bin/python
EXTRA_ARGS=()
if [[ "$PHASE" == full ]]; then EXTRA_ARGS+=(--smoke-dir "$SMOKE_DIR"); fi
if [[ "$PHASE" == smoke ]]; then
    srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,cores --kill-on-bad-exit=1 \
        "$SENSITIVITY_PYTHON" -m pytest -q tests/test_sensitivity_core.py \
        tests/test_sensitivity_audit.py tests/test_inference_sensitivity_runner.py
fi
srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,cores --kill-on-bad-exit=1 \
    "$SENSITIVITY_PYTHON" -m scripts.run_inference_sensitivity \
    --manifest "$MANIFEST" --output-dir "$OUTPUT_DIR" --phase "$PHASE" --group "$GROUP" "${EXTRA_ARGS[@]}"
srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,cores --kill-on-bad-exit=1 \
    "$SENSITIVITY_PYTHON" -m scripts.audit_inference_sensitivity \
    --manifest "$MANIFEST" --output-dir "$OUTPUT_DIR" --phase "$PHASE" --group "$GROUP"
