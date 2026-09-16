from types import SimpleNamespace

import pytest

from src.runtime_cpu import THREAD_VARIABLES, bound_tf_dataset, check_affinity, configure_environment


ENV = {"SLURM_NTASKS": "3", "SLURM_CPUS_PER_TASK": "1", "SLURM_JOB_CPUS_PER_NODE": "3"}


def test_rank_and_batch_limits():
    assert check_affinity(rank=True, environ=ENV, affinity={7})['allowed_cpus'] == [7]
    assert check_affinity(rank=False, environ=ENV, affinity={7, 8, 9})['allocated_cpus'] == 3
    for rank, cpus in [(True, {7, 8}), (False, {1, 2, 3, 4})]:
        with pytest.raises(RuntimeError):
            check_affinity(rank=rank, environ=ENV, affinity=cpus)


@pytest.mark.parametrize('key,value', [('SLURM_NTASKS', '1'), ('SLURM_CPUS_PER_TASK', '4'),
                                       ('SLURM_JOB_CPUS_PER_NODE', '24'), ('SLURM_JOB_CPUS_PER_NODE', '')])
def test_reject_wrong_allocation(key, value):
    with pytest.raises(RuntimeError):
        check_affinity(rank=True, environ={**ENV, key: value}, affinity={0})


def test_environment_is_opt_in_and_overrides_inherited_pools(monkeypatch):
    monkeypatch.delenv('GRID_CPU_BUDGET', raising=False)
    monkeypatch.setenv('OMP_NUM_THREADS', '24')
    configure_environment()
    assert __import__('os').environ['OMP_NUM_THREADS'] == '24'
    monkeypatch.setenv('GRID_CPU_BUDGET', '4')
    for name in THREAD_VARIABLES:
        monkeypatch.setenv(name, '24')
    configure_environment()
    assert all(__import__('os').environ[name] == '1' for name in THREAD_VARIABLES)


def test_tf_dataset_limits_apply_before_first_iteration(monkeypatch):
    monkeypatch.setenv('GRID_CPU_BUDGET', '4')
    options = SimpleNamespace(threading=SimpleNamespace(), autotune=SimpleNamespace())
    tf = SimpleNamespace(data=SimpleNamespace(Options=lambda: options))
    dataset = SimpleNamespace(with_options=lambda x: x)
    result = bound_tf_dataset(dataset, tf)
    assert result.threading.private_threadpool_size == 1
    assert result.threading.max_intra_op_parallelism == 1
    assert result.autotune.enabled is False
