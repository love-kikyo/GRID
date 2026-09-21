import os
from pathlib import Path
import subprocess

import pytest

from src.runtime_export_cpu import check_export_allocation, configure_export_cpu, export_cpu_snapshot

ENV = {'GRID_CPU_BUDGET': '4', 'SLURM_NTASKS': '1', 'SLURM_CPUS_PER_TASK': '4', 'SLURM_JOB_CPUS_PER_NODE': '4'}


def test_export_allocation_accepts_only_single_task_four_cpu_budget():
    assert check_export_allocation(ENV, {2, 3, 8, 9})['allocated_cpus'] == 4
    for field, value in [('SLURM_NTASKS', '3'), ('SLURM_CPUS_PER_TASK', '8'),
                         ('SLURM_JOB_CPUS_PER_NODE', '8'), ('GRID_CPU_BUDGET', '')]:
        with pytest.raises(RuntimeError):
            check_export_allocation({**ENV, field: value}, {2, 3, 8, 9})
    for affinity in (set(), set(range(5))):
        with pytest.raises(RuntimeError):
            check_export_allocation(ENV, affinity)


def test_export_limits_threads_and_workers(monkeypatch):
    calls = []
    class FakeTorch:
        def set_num_threads(self, n): calls.append(('intra', n))
        def set_num_interop_threads(self, n): calls.append(('inter', n))
    for key, value in {**ENV, 'GRID_EXPORT_CPU_BUDGET': '4'}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(os, 'sched_getaffinity', lambda pid: {2, 3, 8, 9})
    assert configure_export_cpu(FakeTorch(), 1)['library_threads'] == 1
    assert calls == [('intra', 1), ('inter', 1)]
    assert os.environ['TF_NUM_INTRAOP_THREADS'] == os.environ['OPENBLAS_NUM_THREADS'] == '1'
    with pytest.raises(ValueError):
        configure_export_cpu(FakeTorch(), 2)
    monkeypatch.delenv('GRID_EXPORT_CPU_BUDGET')
    assert configure_export_cpu(FakeTorch(), 1) is None


def test_process_snapshot_checks_descendant_thread_masks(monkeypatch, tmp_path):
    import src.runtime_export_cpu as runtime
    proc = tmp_path / 'proc'
    pid = os.getpid()
    child = pid + 100000
    for p, parent in [(pid, 0), (child, pid)]:
        (proc / str(p) / 'task' / str(p)).mkdir(parents=True)
        (proc / str(p) / 'stat').write_text(f'{p} (python) S {parent} 0 0')
    def mapped_path(s):
        return Path(str(s).replace('/proc', str(proc), 1))
    monkeypatch.setattr(runtime, 'Path', mapped_path)
    monkeypatch.setattr(os, 'sched_getaffinity', lambda p: {2, 3})
    assert len(export_cpu_snapshot([2, 3])['processes']) == 2
    monkeypatch.setattr(os, 'sched_getaffinity', lambda p: {2, 3, 99} if p == child else {2, 3})
    with pytest.raises(RuntimeError, match='escaped'):
        export_cpu_snapshot([2, 3])


def test_launcher_binds_four_cpus_and_preserves_export_arguments(tmp_path):
    root = Path(__file__).resolve().parents[1]
    run = tmp_path / 'run'
    (run / '.hydra').mkdir(parents=True)
    (run / '.hydra/config.yaml').touch()
    ckpt, preflight = tmp_path / 'model.ckpt', tmp_path / 'preflight.json'
    ckpt.touch(); preflight.touch()
    mocks = '''
srun() { printf 'ARG=%s\\n' "$@"; printf 'THREADS=%s,%s\\n' "$OPENBLAS_NUM_THREADS" "$TF_NUM_INTRAOP_THREADS"; }
mkdir() { :; }
cd() { :; }
'''
    result = subprocess.run(['bash', '-c', mocks + (root / 'run_export_paired_evaluation.sh').read_text(),
                             'export', 'seed43', 'val', str(run), str(ckpt), str(preflight)],
                            env={'PATH': os.defpath, 'USER': 'test', 'SOURCE_GIT_COMMIT': 'unit', 'MAX_BATCHES': '2'},
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    for arg in ['ARG=--cpus-per-task=4', 'ARG=--cpu-bind=verbose,cores', 'ARG=--batch-size\nARG=32',
                'ARG=--num-workers\nARG=1', 'ARG=--max-batches\nARG=2', 'THREADS=1,1']:
        assert arg in result.stdout
