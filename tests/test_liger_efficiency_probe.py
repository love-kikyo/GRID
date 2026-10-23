import copy
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts.accept_liger_efficiency import (
    expected_identity, require_exact_equivalence, summarize_case,
    validate_fixture, validate_memory_snapshot,
)
from scripts.probe_liger_efficiency import capacity_oom, deadline, load_fixture
from src.baselines.liger_data import Record
from src.baselines.liger_prefix_training import prefix_identity_digest
from src.baselines.liger_supervision import expand_training_records, pack_supervised_batch


def test_independent_fixture_identity_matches_real_training_packer():
    rows = [Record(10, tuple(range(12)), 'a' * 64), Record(20, (0, 0, 2), 'b' * 64)]
    examples = expand_training_records(rows)
    sids = np.arange(12 * 4).reshape(12, 4)
    content = np.arange(12 * 3).reshape(12, 3)
    for start in range(0, len(examples), 4):
        batch = examples[start:start + 4]
        _, meta = pack_supervised_batch(batch, sids, content)
        assert expected_identity(batch) == prefix_identity_digest(meta)
    assert expected_identity(examples) != expected_identity(list(reversed(examples)))


def reports():
    a = dict(status='passed', backend='lazy', batch_size=1024, gpu={'uuid': 'same'},
        initial_state_sha256='init', learning_state_sha256='all_state',
        fixture_receipt={'sha256': 'fixture'}, final_progress={'updates': 4},
        steps=[dict(training={'sid_loss': 2., 'prefix_identity_sha256': str(i)}, seconds=i + 1) for i in range(4)])
    b = copy.deepcopy(a)
    b['backend'] = 'cached_seen'
    b['steps'][1]['seconds'] = 100
    return a, b


def test_exact_equivalence_ignores_timing_but_not_learning():
    require_exact_equivalence(*reports())


@pytest.mark.parametrize('change', ['status', 'gpu', 'init', 'state', 'progress', 'loss', 'identity', 'fixture'])
def test_equivalence_fails_without_widening_tolerance(change):
    a, b = reports()
    if change == 'status': b['status'] = 'cuda_oom'
    elif change == 'gpu': b['gpu']['uuid'] = 'different'
    elif change == 'init': b['initial_state_sha256'] = 'changed'
    elif change == 'state': b['learning_state_sha256'] = 'changed'
    elif change == 'progress': b['final_progress']['updates'] = 3
    elif change == 'loss': b['steps'][0]['training']['sid_loss'] += 1e-12
    elif change == 'identity': b['steps'][0]['training']['prefix_identity_sha256'] = 'changed'
    else: b['fixture_receipt']['sha256'] = 'changed'
    with pytest.raises(AssertionError):
        require_exact_equivalence(a, b)


def test_summary_uses_three_predeclared_warm_measurements_and_exact_tail():
    steps = [dict(measured=i > 0, training_wall_seconds=t,
        catalog_fetch={'cpu_fetch_wall_seconds': 1.}, model_forward={'model_forward_wall_seconds': .5},
        peak_gpu_allocated_bytes=100 + i) for i, t in enumerate((1000., 8., 10., 12.))]
    r = dict(name='cached2048', batch_size=2048, steps=steps, cache_build_seconds=30.,
        cache_verify_seconds=20., setup_seconds=40., memory=[{'high_water_rss_bytes': 999}])
    s = summarize_case(r, 45996113)
    assert s['mean_training_wall_seconds'] == 10.
    assert s['prefix_examples_per_second'] == 204.8
    compute = math.ceil(45996113 / 2048) * 10
    assert s['indicative_pass_compute_hours'] == compute / 3600
    assert s['indicative_pass_compute_plus_cache_build_hours'] == (compute + 30) / 3600
    assert s['indicative_pass_compute_plus_setup_and_cache_hours'] == (compute + 90) / 3600
    assert s['one_time_setup_and_cache_seconds'] == 90.
    assert s['max_process_self_hwm_bytes'] == 999
    assert 'excludes streaming' in s['scope'] and s['max_gpu_allocated_bytes'] == 103
    r['steps'].pop()
    with pytest.raises(AssertionError):
        summarize_case(r, 45996113)


def test_deadline_is_shared_across_case_boundaries(monkeypatch):
    monkeypatch.setattr('scripts.probe_liger_efficiency.time.time', lambda: 100.)
    deadline({'deadline_unix_seconds': 101.})
    with pytest.raises(TimeoutError):
        deadline({'deadline_unix_seconds': 100.})


def test_fixture_rejects_another_job_before_opening_payload(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', 'new-job')
    (tmp_path / 'fixture_receipt.json').write_text(json.dumps({'status': 'passed', 'job_id': 'old-job'}))
    with pytest.raises(AssertionError):
        load_fixture(tmp_path / 'manifest.json', {})


def test_oom_is_limited_to_explicit_cuda_failure_and_skips_larger_case():
    source = (Path(__file__).resolve().parents[1] / 'scripts/probe_liger_efficiency.py').read_text()
    assert 'except torch.cuda.OutOfMemoryError:' in source
    assert "previous['status'] == 'cuda_oom'" in source
    assert "status='skipped_after_cuda_oom'" in source
    assert "capacity_oom(stage, setting['batch_size'], current_attempt)" in source
    assert 'if not is_capacity:' in source
    assert 'except Exception:' in source and "report['status'] = 'failed'" in source
    assert 'torch.cuda.empty_cache()' not in source


@pytest.mark.parametrize('stage,size,attempt,expected', [
    ('training_attempt', 2048, 1, True), ('training_attempt', 4096, 4, True),
    ('training_attempt', 1024, 1, False), ('training_attempt', 2048, None, False),
    ('training_attempt', 2048, True, False), ('training_attempt', 2048, 5, False),
    ('runtime_initialization', 2048, None, False), ('cache_build', 4096, None, False),
    ('cache_verification', 4096, None, False), ('before_training_attempt', 2048, 2, False),
    ('post_update_checks', 4096, 4, False), ('state_fingerprint', 4096, 4, False),
])
def test_only_large_batch_training_oom_counts_as_capacity(stage, size, attempt, expected):
    assert capacity_oom(stage, size, attempt) is expected


def memory():
    return dict(schema='liger.process_self_resources.v1', errors=[], rss_bytes=100,
        high_water_rss_bytes=120, virtual_memory_bytes=200, anonymous_rss_bytes=60,
        file_rss_bytes=40, shared_rss_bytes=0, swap_bytes=0, ru_maxrss_bytes=121,
        threads=1, clock_ticks_per_second=100, monotonic_seconds=1.,
        proc_cpu_user_seconds=.2, proc_cpu_system_seconds=.3, proc_cpu_total_seconds=.5,
        resource_cpu_user_seconds=.21, resource_cpu_system_seconds=.31)


def test_memory_checks_are_process_self_not_a_job_memory_oracle():
    value = memory()
    validate_memory_snapshot(value)
    value.update(high_water_rss_bytes=25 * 1024**3, ru_maxrss_bytes=26 * 1024**3)
    validate_memory_snapshot(value)


@pytest.mark.parametrize('key,value', [
    ('rss_bytes', None), ('rss_bytes', 0), ('high_water_rss_bytes', 99),
    ('virtual_memory_bytes', 99), ('errors', ['proc_status:OSError']),
    ('swap_bytes', -1), ('threads', 0), ('proc_cpu_total_seconds', .9),
    ('resource_cpu_user_seconds', float('nan')), ('ru_maxrss_bytes', True),
])
def test_bad_resource_snapshots_are_rejected(key, value):
    snapshot = memory()
    snapshot[key] = value
    with pytest.raises(AssertionError):
        validate_memory_snapshot(snapshot)


def fixture_values():
    examples = expand_training_records([
        Record(10, tuple(range(12)), 'a' * 64), Record(20, (0, 0, 2), 'b' * 64)])
    receipt = dict(job_id='same', prefix_examples=9, unique_parents=2, final_target_count=2,
        data_batches=[dict(index=i, rows=3, seconds=.1) for i in range(3)],
        started_unix_seconds=100., deadline_unix_seconds=2500., elapsed_seconds=1.,
        formal_training_ready=False, automatic_successor=False,
        validation_test_records_read=False, formal_checkpoint_produced=False,
        resources=dict(allocated_cpus=4, user_cpu_ceiling=4, library_threads=1,
            job_id='same', allowed_cpus=[0, 1, 2, 3]),
        cpu_checks=[dict(allowed_cpus=[0, 1, 2, 3], processes=[dict(pid=100, threads_checked=3)]) for _ in range(2)],
        memory=memory())
    return receipt, examples


def test_fixture_recomputes_counts_structure_and_resources():
    receipt, examples = fixture_values()
    validate_fixture(receipt, examples, asset_rows=12, expected_prefixes=9, batch_size=3)


@pytest.mark.parametrize('change', [
    'duplicate', 'wrong_parent', 'window', 'prefix_position', 'final', 'prefix_count',
    'invalid_row', 'conflicting_history', 'unique_parents', 'final_count', 'batch_rows',
    'batch_index', 'seconds', 'deadline', 'cpu', 'cpu_checks', 'memory', 'heldout',
])
def test_fixture_rejects_bad_identity_counts_timing_or_resources(change):
    receipt, examples = fixture_values()
    if change == 'duplicate': examples[1] = examples[0]
    elif change == 'wrong_parent': examples[0] = replace(examples[0], parent_serialized_sha256='c' * 64)
    elif change == 'window': examples[0] = replace(examples[0], window_start_index=0)
    elif change == 'prefix_position': examples[0] = replace(examples[0], prefix_target_index=0)
    elif change == 'final': examples[0] = replace(examples[0], is_final=True)
    elif change == 'prefix_count': examples[0] = replace(examples[0], parent_prefix_count=1)
    elif change == 'invalid_row': examples[0] = replace(examples[0], sequence=(12, 5))
    elif change == 'conflicting_history': examples[0] = replace(examples[0], sequence=(0, 5))
    elif change == 'unique_parents': receipt['unique_parents'] = 9
    elif change == 'final_count': receipt['final_target_count'] = 0
    elif change == 'batch_rows': receipt['data_batches'][0]['rows'] = 2
    elif change == 'batch_index': receipt['data_batches'][1]['index'] = 0
    elif change == 'seconds': receipt['data_batches'][0]['seconds'] = float('nan')
    elif change == 'deadline': receipt['deadline_unix_seconds'] = 2600.
    elif change == 'cpu': receipt['resources']['allocated_cpus'] = 24
    elif change == 'cpu_checks': receipt['cpu_checks'].pop()
    elif change == 'memory': receipt['memory']['errors'] = ['unavailable']
    else: receipt['validation_test_records_read'] = True
    with pytest.raises(AssertionError):
        validate_fixture(receipt, examples, asset_rows=12, expected_prefixes=9, batch_size=3)
