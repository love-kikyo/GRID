#!/bin/bash
#SBATCH --job-name=RetrContent-Multi
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
ASSET_ROOT=/mnt/beegfs/home/MMReco2021/liuyu/GRID
if [[ $# -ne 1 ]]; then echo 'Usage: ABSOLUTE_SEED43_OR_44_MANIFEST'; exit 2; fi
MANIFEST=$1
case "$MANIFEST" in
  "$ASSET_ROOT"/logs/training_launch/retrieval_content_seed4[34]_full_*/manifest.json) ;;
  *) echo 'Require an absolute seed43/44 retrieval-content manifest path'; exit 2 ;;
esac
if [[ "$MANIFEST" == *'/../'* || "$MANIFEST" == *'/./'* || "$MANIFEST" == *'//'* ]]; then
    echo 'Require a canonical manifest path'; exit 2
fi
LAUNCH_DIR=${MANIFEST%/manifest.json}
LAUNCH_NAME=${LAUNCH_DIR##*/}
OUTPUT_ROOT="$ASSET_ROOT/logs/analysis/$LAUNCH_NAME"
if [[ -e "$OUTPUT_ROOT" ]]; then echo 'Refusing an existing experiment output root'; exit 2; fi
if [[ "${SLURM_CPUS_PER_TASK:-}" != 4 || "${SLURM_NTASKS:-}" != 1 ]]; then
    echo 'Require a single-task, four-CPU allocation'; exit 2
fi
: "${SLURM_JOB_ID:?Required Slurm job ID}"
: "${SOURCE_GIT_COMMIT:?Required execution commit}"
: "${CONTENT_MANIFEST_SHA256:?Required content manifest hash}"
cd /mnt/beegfs/home/MMReco2021/liuyu/GRID-retrieval-content-multiseed
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false USE_TF=0
export GRID_EXPORT_CPU_BUDGET=4 GRID_CPU_BUDGET=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/grid-content-multiseed-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
CONTENT_PYTHON=/home/MMReco2021/.conda/envs/GRID/bin/python
run_bound() {
    srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,cores --kill-on-bad-exit=1 \
        "$CONTENT_PYTHON" "$@"
}
# Every seed proves its own checkpoint/inputs and five-policy GPU smoke. A
# failed test, artifact audit, or statistics step stops this job before full.
run_bound -m pytest -q tests/test_retrieval_content_manifest.py \
    tests/test_retrieval_content_core.py tests/test_retrieval_content_runner.py \
    tests/test_retrieval_content_audit.py tests/test_retrieval_content_paired.py \
    tests/test_retrieval_paired_report_comparison.py tests/test_worktree_provenance.py \
    tests/test_retrieval_content_multiseed_launcher.py
run_bound -m scripts.run_retrieval_content --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_ROOT/smoke" --phase smoke
run_bound -m scripts.audit_retrieval_content --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_ROOT/smoke" --phase smoke
run_bound -m scripts.analyze_retrieval_content --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_ROOT/smoke" --phase smoke
run_bound -m scripts.run_retrieval_content --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_ROOT/full" --phase full --smoke-dir "$OUTPUT_ROOT/smoke"
run_bound -m scripts.audit_retrieval_content --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_ROOT/full" --phase full
run_bound -m scripts.analyze_retrieval_content --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_ROOT/full" --phase full
