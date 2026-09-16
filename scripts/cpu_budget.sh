# Source only after GRID_ROOT and PYTHON_BIN are set. Applies before any ML import.
export GRID_CPU_BUDGET=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
if [[ "${SLURM_CPUS_PER_TASK:-}" != 1 || "${SLURM_NTASKS:-}" != 3 || "${SLURM_NTASKS_PER_NODE:-}" != 3 ]]; then
    echo 'Expected a single-node, three-task Slurm allocation with one CPU per task.' >&2
    exit 1
fi
"${PYTHON_BIN}" -m src.runtime_cpu
