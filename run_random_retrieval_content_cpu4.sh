#!/bin/bash
# One allocation: own full validation/selection -> audited smoke -> full test.
# Every phase must pass before the next; no training and no old beta reuse.
#SBATCH --job-name=Random42-RetrContent
#SBATCH --output=/mnt/beegfs/home/MMReco2021/liuyu/GRID/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/beegfs/home/MMReco2021/liuyu/GRID/logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:4090:1
#SBATCH --partition=rtx4090
#SBATCH --time=06:00:00
set -euo pipefail
GRID_ROOT=/mnt/beegfs/home/MMReco2021/liuyu/GRID
if [[ $# -ne 1 || ! "$1" =~ ^random_init_retrieval_seed42_158562_[0-9]{4}-[0-9]{2}-[0-9]{2}_v[0-9]+$ ]]; then
    echo 'Require a fresh random_init_retrieval_seed42_158562_YYYY-MM-DD_vN experiment name'; exit 2
fi
if ! [[ "${SOURCE_GIT_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]]; then
    echo 'Require explicit SOURCE_GIT_COMMIT'; exit 2
fi
if [[ "${SLURM_CPUS_PER_TASK:-}" != 4 || "${SLURM_NTASKS:-}" != 1 ]]; then
    echo 'Require one Slurm task with exactly four CPUs'; exit 2
fi
: "${SLURM_JOB_ID:?Required Slurm job ID}"
cd "$GRID_ROOT"
OUTPUT_ROOT="$GRID_ROOT/logs/analysis/$1"
LAUNCH_ROOT="$GRID_ROOT/logs/training_launch/$1"
MANIFEST="$LAUNCH_ROOT/manifest.json"
if [[ -e "$OUTPUT_ROOT" || -e "$MANIFEST" ]]; then
    echo 'Refusing existing output or manifest'; exit 2
fi
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false USE_TF=0
export GRID_EXPORT_CPU_BUDGET=4 GRID_FUSION_CPU_BUDGET=4 GRID_CPU_BUDGET=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/grid-random-content-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$LAUNCH_ROOT"
CONTENT_PYTHON=/home/MMReco2021/.conda/envs/GRID/bin/python
git diff --exit-code "$SOURCE_GIT_COMMIT" -- src scripts tests configs run_random_retrieval_content_cpu4.sh
if [[ -n "$(git status --porcelain -- src scripts tests configs run_random_retrieval_content_cpu4.sh)" ]]; then
    echo 'Execution sources must be committed and unchanged'; exit 2
fi
run_bound() {
    srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,cores --kill-on-bad-exit=1 \
        "$CONTENT_PYTHON" "$@"
}
run_bound -m pytest -q tests/test_random_retrieval_validation.py \
    tests/test_random_retrieval_manifest.py tests/test_retrieval_diagnostic_core.py \
    tests/test_retrieval_content_core.py tests/test_retrieval_content_audit.py \
    tests/test_retrieval_content_paired.py tests/test_retrieval_paired_report_comparison.py
run_bound -m scripts.random_retrieval_validation --output-dir "$OUTPUT_ROOT/validation" \
    --expected-commit "$SOURCE_GIT_COMMIT"
run_bound -m scripts.prepare_random_retrieval_content --manifest "$MANIFEST" \
    --validation-dir "$OUTPUT_ROOT/validation" --expected-commit "$SOURCE_GIT_COMMIT"
CONTENT_MANIFEST_SHA256=$(sha256sum "$MANIFEST")
export CONTENT_MANIFEST_SHA256=${CONTENT_MANIFEST_SHA256%% *}
run_bound -m scripts.run_random_retrieval_content --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_ROOT/smoke" --phase smoke
run_bound -m scripts.audit_retrieval_content --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_ROOT/smoke" --phase smoke
run_bound -m scripts.analyze_retrieval_content --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_ROOT/smoke" --phase smoke
run_bound -m scripts.run_random_retrieval_content --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_ROOT/full" --phase full --smoke-dir "$OUTPUT_ROOT/smoke"
run_bound -m scripts.audit_retrieval_content --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_ROOT/full" --phase full
run_bound -m scripts.analyze_retrieval_content --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_ROOT/full" --phase full
echo 'Random-init retrieval-content pipeline completed; full acceptance and paired report written.'
