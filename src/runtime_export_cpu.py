"""Four-core single-process export budget; no model or inference changes."""
import os
from pathlib import Path

from src.runtime_cpu import configure_environment, configure_torch


def check_export_allocation(environ=None, affinity=None):
    env = os.environ if environ is None else environ
    cpus = sorted(os.sched_getaffinity(0) if affinity is None else affinity)
    if (env.get('GRID_CPU_BUDGET') != '4' or env.get('SLURM_NTASKS') != '1'
            or env.get('SLURM_CPUS_PER_TASK') != '4'
            or env.get('SLURM_JOB_CPUS_PER_NODE') not in {'4', '4(x1)'}):
        raise RuntimeError('Export requires one task and four CPUs on one node')
    if not cpus or len(cpus) > 4:
        raise RuntimeError(f'Export CPU affinity exceeds four-core ceiling: {cpus}')
    return {'allocated_cpus': 4, 'user_cpu_ceiling': 4, 'allowed_cpus': cpus,
            'job_id': env.get('SLURM_JOB_ID'), 'library_threads': 1}


def configure_export_cpu(torch, num_workers):
    if os.environ.get('GRID_EXPORT_CPU_BUDGET') != '4':
        return None
    report = check_export_allocation()
    if num_workers not in (0, 1):
        raise ValueError('Bounded export supports at most one data worker')
    configure_environment()
    configure_torch(torch)
    return report


def export_cpu_snapshot(allowed_cpus):
    """Check all live threads of this exporter and its descendant processes."""
    allowed = set(allowed_cpus)
    if not allowed or len(allowed) > 4 or set(os.sched_getaffinity(0)) != allowed:
        raise RuntimeError('Export affinity changed since startup')
    parents = {}
    for stat in Path('/proc').glob('[0-9]*/stat'):
        try:
            parents[int(stat.parent.name)] = int(stat.read_text().rpartition(')')[2].split()[1])
        except (OSError, ValueError, IndexError):
            continue
    selected = {os.getpid()}
    while True:
        added = {pid for pid, parent in parents.items() if parent in selected} - selected
        if not added:
            break
        selected.update(added)
    processes = []
    for pid in sorted(selected):
        count = 0
        for task in Path(f'/proc/{pid}/task').glob('[0-9]*'):
            try:
                mask = set(os.sched_getaffinity(int(task.name)))
            except ProcessLookupError:
                continue
            if not mask or not mask <= allowed:
                raise RuntimeError(f'Export descendant thread {task.name} escaped CPU budget')
            count += 1
        if count:
            processes.append({'pid': pid, 'threads_checked': count})
    return {'allowed_cpus': sorted(allowed), 'processes': processes}
