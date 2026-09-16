"""Opt-in CPU budget for shared-node training; safe to import before ML libraries."""

import json
import os
from pathlib import Path
import time


THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS", "VECLIB_MAXIMUM_THREADS",
    "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS",
)


def enabled():
    return os.environ.get("GRID_CPU_BUDGET") == "4"


def configure_environment():
    if enabled():
        for name in THREAD_VARIABLES:
            os.environ[name] = "1"
        os.environ["OMP_DYNAMIC"] = "FALSE"
        os.environ["MKL_DYNAMIC"] = "FALSE"


def check_affinity(*, rank, environ=None, affinity=None):
    env = os.environ if environ is None else environ
    cpus = sorted(os.sched_getaffinity(0) if affinity is None else affinity)
    if env.get("SLURM_NTASKS") != "3" or env.get("SLURM_CPUS_PER_TASK") != "1":
        raise RuntimeError("CPU budget requires exactly three Slurm tasks with one CPU each")
    # This launcher is deliberately single-node; reject expanded/ambiguous allocations.
    if env.get("SLURM_JOB_CPUS_PER_NODE") not in {"3", "3(x1)"}:
        raise RuntimeError("Expected exactly three allocated CPUs on one node")
    expected = 1 if rank else 3
    if not cpus or len(cpus) > expected:
        raise RuntimeError(f"CPU affinity exceeds {'rank' if rank else 'job'} budget: {cpus}")
    return {"job_id": env.get("SLURM_JOB_ID"), "rank": env.get("SLURM_PROCID") if rank else None,
            "allowed_cpus": cpus, "allocated_cpus": 3, "user_cpu_ceiling": 4}


def configure_torch(torch):
    if enabled():
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)


def bound_tf_dataset(dataset, tf):
    if not enabled():
        return dataset
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    options.autotune.enabled = False
    return dataset.with_options(options)


def process_tree_snapshot():
    """Inspect only this rank and its descendants, including every live thread."""
    records = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rpartition(')')[2].split()
            records[int(path.parent.name)] = (int(fields[1]), int(fields[11]) + int(fields[12]))
        except (OSError, ValueError, IndexError):
            continue
    selected = {os.getpid()}
    while True:
        added = {pid for pid, (parent, _) in records.items() if parent in selected} - selected
        if not added:
            break
        selected.update(added)
    processes, allowed = [], set()
    for pid in sorted(selected):
        masks = []
        for task in Path(f'/proc/{pid}/task').glob('[0-9]*'):
            try:
                cpus = os.sched_getaffinity(int(task.name))
            except ProcessLookupError:
                continue
            if len(cpus) != 1:
                raise RuntimeError(f'Rank descendant thread {task.name} escaped one-CPU binding: {sorted(cpus)}')
            allowed.update(cpus)
            masks.append(sorted(cpus))
        if masks:
            processes.append({'pid': pid, 'threads': len(masks),
                              'cpu_seconds': records.get(pid, (0, 0))[1] / os.sysconf('SC_CLK_TCK')})
    if allowed != set(os.sched_getaffinity(0)):
        raise RuntimeError(f'Rank process tree spans extra CPUs: {sorted(allowed)}')
    return {'monotonic_seconds': time.monotonic(), 'allowed_cpus': sorted(allowed),
            'processes': processes, 'cpu_seconds': sum(x['cpu_seconds'] for x in processes)}


if __name__ == "__main__":
    configure_environment()
    print("[cpu-budget] " + json.dumps(check_affinity(rank=False)), flush=True)
