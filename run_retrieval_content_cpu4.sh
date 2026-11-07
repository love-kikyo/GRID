#!/bin/bash
#SBATCH --job-name=RetrContent42
#SBATCH --output=/mnt/beegfs/home/MMReco2021/liuyu/GRID/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/beegfs/home/MMReco2021/liuyu/GRID/logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:4090:1
#SBATCH --partition=rtx4090
#SBATCH --nodelist=aias-compute-2
#SBATCH --time=06:00:00
set -euo pipefail
if [[ $# -ne 4 ]]; then echo 'Usage: MANIFEST OUT {smoke|full} SMOKE_DIR_OR_DASH'; exit 2; fi
MANIFEST=$1
OUTPUT_DIR=$2
PHASE=$3
SMOKE_DIR=$4
if [[ "$MANIFEST" != /* || "$OUTPUT_DIR" != /* ]]; then echo 'Require absolute paths'; exit 2; fi
if [[ "$PHASE" != smoke && "$PHASE" != full ]]; then echo 'Unsupported phase'; exit 2; fi
if [[ "$PHASE" == full && "$SMOKE_DIR" != /* ]]; then echo 'Require accepted smoke directory'; exit 2; fi
if [[ "$PHASE" == smoke && "$SMOKE_DIR" != - ]]; then echo 'Smoke cannot reuse prior gate'; exit 2; fi
cd /mnt/beegfs/home/MMReco2021/liuyu/GRID-sensitivity-multiseed
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false USE_TF=0
export GRID_EXPORT_CPU_BUDGET=4 GRID_CPU_BUDGET=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export MPLCONFIGDIR="/tmp/grid-retrieval-content-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
: "${SOURCE_GIT_COMMIT:?Required execution commit}"
: "${CONTENT_MANIFEST_SHA256:?Required content manifest hash}"
CONTENT_PYTHON=/home/MMReco2021/.conda/envs/GRID/bin/python
EXTRA_ARGS=()
if [[ "$PHASE" == full ]]; then EXTRA_ARGS+=(--smoke-dir "$SMOKE_DIR"); fi
if [[ "$PHASE" == smoke ]]; then
    srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,cores --kill-on-bad-exit=1 \
        "$CONTENT_PYTHON" -m pytest -q tests/test_retrieval_content_manifest.py \
        tests/test_retrieval_content_core.py tests/test_retrieval_content_runner.py \
        tests/test_retrieval_content_audit.py tests/test_retrieval_content_paired.py
fi
srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,cores --kill-on-bad-exit=1 \
    "$CONTENT_PYTHON" -m scripts.run_retrieval_content \
    --manifest "$MANIFEST" --output-dir "$OUTPUT_DIR" --phase "$PHASE" "${EXTRA_ARGS[@]}"
srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,cores --kill-on-bad-exit=1 \
    "$CONTENT_PYTHON" -m scripts.audit_retrieval_content \
    --manifest "$MANIFEST" --output-dir "$OUTPUT_DIR" --phase "$PHASE"
srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,cores --kill-on-bad-exit=1 \
    "$CONTENT_PYTHON" -m scripts.analyze_retrieval_content \
    --manifest "$MANIFEST" --output-dir "$OUTPUT_DIR" --phase "$PHASE"
