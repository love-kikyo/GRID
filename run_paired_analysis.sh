#!/bin/bash
# Analyze one controlled SID/listwise/pointwise/separate set after each method has
# a full test export and validation-only fusion selection.
# Usage:
#   sbatch run_paired_analysis.sh OUTPUT_TAG \
#     LISTWISE_TEST LISTWISE_SELECTION \
#     POINTWISE_TEST POINTWISE_SELECTION \
#     SEPARATE_TEST SEPARATE_SELECTION SID_TEST

#SBATCH --job-name=GRID-PairedAnalysis
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:0

set -euo pipefail

if [[ "$#" -ne 8 ]]; then
    echo "Usage: sbatch $0 OUTPUT_TAG LISTWISE_TEST LISTWISE_SELECTION POINTWISE_TEST POINTWISE_SELECTION SEPARATE_TEST SEPARATE_SELECTION SID_TEST" >&2
    exit 2
fi

OUTPUT_TAG="$1"
LISTWISE_TEST="$2"
LISTWISE_SELECTION="$3"
POINTWISE_TEST="$4"
POINTWISE_SELECTION="$5"
SEPARATE_TEST="$6"
SEPARATE_SELECTION="$7"
SID_TEST="$8"

if ! [[ "${OUTPUT_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "OUTPUT_TAG may contain only letters, digits, dot, underscore, and hyphen." >&2
    exit 2
fi
for path_argument in \
    "${LISTWISE_TEST}" "${LISTWISE_SELECTION}" \
    "${POINTWISE_TEST}" "${POINTWISE_SELECTION}" \
    "${SEPARATE_TEST}" "${SEPARATE_SELECTION}" "${SID_TEST}"; do
    if [[ "${path_argument}" != /* ]]; then
        echo "All export and selection paths must be absolute: ${path_argument}" >&2
        exit 2
    fi
done
for required in \
    "${LISTWISE_TEST}" "${LISTWISE_TEST%.h5}.json" "${LISTWISE_SELECTION}" \
    "${POINTWISE_TEST}" "${POINTWISE_TEST%.h5}.json" "${POINTWISE_SELECTION}" \
    "${SEPARATE_TEST}" "${SEPARATE_TEST%.h5}.json" "${SEPARATE_SELECTION}" \
    "${SID_TEST}" "${SID_TEST%.h5}.json"; do
    if [[ ! -f "${required}" ]]; then
        echo "Missing required export/provenance file: ${required}" >&2
        exit 2
    fi
done

GRID_ROOT=/mnt/beegfs/home/MMReco2021/liuyu/GRID
PYTHON_BIN=/home/MMReco2021/.conda/envs/GRID/bin/python
ANALYSIS_DIR=${GRID_ROOT}/logs/analysis/paired_evaluation

cd "${GRID_ROOT}"
mkdir -p logs/slurm "${ANALYSIS_DIR}"
export PYTHONUNBUFFERED=1
export GRID_CPU_BUDGET=4 GRID_PAIRED_CPU_BUDGET=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE

srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,cores --kill-on-bad-exit=1 \
    "${PYTHON_BIN}" -m scripts.analyze_paired_cpu4 \
    --input "sid=${SID_TEST}" \
    --input "listwise=${LISTWISE_TEST}" \
    --input "pointwise=${POINTWISE_TEST}" \
    --input "separate=${SEPARATE_TEST}" \
    --selection "listwise=${LISTWISE_SELECTION}" \
    --selection "pointwise=${POINTWISE_SELECTION}" \
    --selection "separate=${SEPARATE_SELECTION}" \
    --compare "sid:beam,listwise:selected" \
    --compare "sid:beam,pointwise:selected" \
    --compare "sid:beam,separate:selected" \
    --compare "pointwise:selected,listwise:selected" \
    --compare "separate:selected,listwise:selected" \
    --compare "listwise:beam,listwise:selected" \
    --compare "pointwise:beam,pointwise:selected" \
    --compare "separate:beam,separate:selected" \
    --compare "listwise:click,listwise:selected" \
    --compare "pointwise:click,pointwise:selected" \
    --compare "separate:click,separate:selected" \
    --ks 5 10 \
    --bootstrap-samples 5000 \
    --bootstrap-batch-size 500 \
    --confidence 0.95 \
    --seed 2027 \
    --expected-split test \
    --output-json "${ANALYSIS_DIR}/paired_report_${OUTPUT_TAG}.json" \
    --output-md "${ANALYSIS_DIR}/paired_report_${OUTPUT_TAG}.md"
