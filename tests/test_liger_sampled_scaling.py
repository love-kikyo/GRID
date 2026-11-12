"""CPU-only synthetic acceptance checks; no production data or CUDA reads."""
import copy
from functools import lru_cache
import math
import statistics

import pytest

from scripts import accept_liger_sampled_scaling as gate
from scripts import probe_liger_sampled_scaling as probe
from src.baselines.liger_data import Record
from src.baselines.liger_prefix_training import initial_progress
from src.baselines.liger_sampled_catalog import sampling_binding
from src.baselines.liger_supervision import expand_training_records


@pytest.mark.parametrize('stage,size,attempt,expected', [
    ('training_attempt', 2048, 1, True), ('training_attempt', 4096, 20, True),
    ('training_attempt', 1024, 1, False), ('training_attempt', 8192, 1, False),
    ('training_attempt', 2048, True, False), ('training_attempt', 2048, None, False),
    ('training_attempt', 2048, 21, False), ('training_attempt', 2048, 0, False),
    ('cache_build', 4096, 1, False), ('cache_verification', 4096, 1, False),
    ('runtime_initialization', 2048, 1, False), ('post_update_checks', 2048, 1, False),
])
def test_capacity_requires_a_real_large_batch_training_stage(stage, size, attempt, expected):
    assert probe.capacity_oom(stage, size, attempt) is expected


def test_deadline_is_shared_across_backend_and_batch_cases(monkeypatch):
    monkeypatch.setattr('scripts.probe_liger_sampled.time.time', lambda: 100.)
    probe.deadline({'deadline_unix_seconds': 101.})
    with pytest.raises(TimeoutError):
        probe.deadline({'deadline_unix_seconds': 100.})


@lru_cache()
def examples():
    return tuple(expand_training_records([
        Record(i, tuple(range(8)), f'{i:064x}') for i in range(11703)
    ])[:81920])


def counter(calls, rows, wall, cpu):
    return dict(calls=calls, rows=rows, wall_seconds=wall, cpu_seconds=cpu,
                failed_calls=0, unknown_rows_calls=0, timing_scope='synthetic CPU scope')


def io_delta():
    return dict(schema='liger.process_access_io_delta.v1', pid=123,
        wall_seconds=1., cpu_seconds=.3, minor_faults=7, major_faults=0,
        io={key: 0 for key in ('rchar', 'wchar', 'syscr', 'syscw', 'read_bytes',
                              'write_bytes', 'cancelled_write_bytes')},
        unavailable_fields=[], counter_reset_fields=[], errors=[],
        timing_scope='zero counters do not prove no storage access')


def memory(stage):
    return dict(stage=stage, schema='liger.process_self_resources.v1', errors=[], rss_bytes=100,
        high_water_rss_bytes=120, virtual_memory_bytes=200, anonymous_rss_bytes=60,
        file_rss_bytes=40, shared_rss_bytes=0, swap_bytes=0, ru_maxrss_bytes=121,
        threads=1, clock_ticks_per_second=100, monotonic_seconds=1.,
        proc_cpu_user_seconds=.2, proc_cpu_system_seconds=.3, proc_cpu_total_seconds=.5,
        resource_cpu_user_seconds=.21, resource_cpu_system_seconds=.31)


@lru_cache()
def report_template(size=1024, backend='lazy', negatives=8192):
    suffix = '8k' if negatives == 8192 else '32k'
    setting = dict(name=f"{'lazy' if backend == 'lazy' else 'cached'}{suffix}{size}",
                   backend=backend, batch_size=size, num_negatives=negatives, seed=42)
    objective = sampling_binding(dict(num_negatives=negatives, seed=42))
    fixture_entry = dict(path='/fixture.json', sha256='f' * 64)
    binding = dict(dense_objective=objective, probe_case=setting['name'],
        probe_batch_size=size, feature_backend=backend, probe_fixture_sha256=fixture_entry['sha256'],
        standardization_stats_sha256='1' * 64, standardization_seen_rows_sha256='2' * 64,
        assets={'model_embedding_copy': '3' * 64})
    steps = []
    for index in range(20):
        batch = examples()[index * size:(index + 1) * size]
        forced = len({e.sequence[-1] for e in batch})
        hist = sum(len(e.sequence[:-1][-7:]) for e in batch)
        identity = gate.expected_identity(batch)
        training = dict(batch=index + 1, rows=size, attempt=index + 1, prefix_count=size,
            final_target_count=sum(e.is_final for e in batch),
            batch_unique_parents=len({e.parent_user_id for e in batch}),
            batch_identity_sha256=identity, prefix_identity_sha256=identity,
            first_user=batch[0].parent_user_id, last_user=batch[-1].parent_user_id,
            updates=index + 1, skipped=False, learning_rate=.0003 * (index + 1) / 963,
            scale_before=65536., scale_after=65536., sid_loss=2., dense_loss=3., total_loss=5.,
            gradient_tensors=152, grad_norm=1., dense_objective=objective,
            forced_positive_classes=forced, sampled_negative_classes=negatives,
            dense_candidate_classes=negatives + forced, sampling_attempt=index)
        catalog = dict(calls=1, rows=negatives + forced, failed_calls=0,
                       unknown_rows_calls=0, cpu_fetch_wall_seconds=.2)
        lazy = backend == 'lazy'
        raw = counter(size + 1, hist + negatives + forced, .2, .1) if lazy else counter(0, 0, 0., 0.)
        std = counter(size + 1, hist + negatives + forced, .5, .3) if lazy else counter(0, 0, 0., 0.)
        access = dict(raw=raw, standardized=std, history=counter(size, hist, .4, .2),
            catalog=catalog, standardization_overhead_wall_seconds=.3 if lazy else 0.,
            standardization_overhead_cpu_seconds=.2 if lazy else 0., timing_scope='nested CPU scopes')
        steps.append(dict(training=training, measured=index >= 5,
            training_wall_seconds=1000. if index < 5 else float(index - 4),
            catalog_fetch=catalog, access=access, io_delta=io_delta(),
            peak_gpu_allocated_bytes=100 + index, peak_gpu_reserved_bytes=200 + index))
    p = initial_progress(max_optimizer_steps=19260, max_attempts=20)
    p.update(next_batch=20, samples_in_epoch=20 * size, batches_consumed=20, attempts=20,
        prefix_count=20 * size, final_count=sum(e.is_final for e in examples()[:20 * size]),
        updates=20, dense_objective=objective)
    manifest = dict(source_git_commit='c' * 40, phase='sampled_scaling_fixture',
                    counts={'prefix_examples_per_pass': 45996113})
    config = {'training': dict(max_optimizer_steps=19260, max_attempts=20, batch_size=1024)}
    report = dict(status='passed', **setting, **manifest, job_id='job', manifest_sha256='d' * 64,
        fixture_receipt=fixture_entry, steps=steps, elapsed_seconds=30.,
        gpu=dict(name='RTX 4090', uuid='same', total_bytes=24 * 1024**3),
        versions={'torch': 'fixture'}, initial_state_sha256='a' * 64,
        learning_state_sha256='b' * 64, final_progress=p, checkpoint_binding=binding,
        setup_seconds=20., cache_build_seconds=0. if backend == 'lazy' else 30.,
        cache_verify_seconds=0. if backend == 'lazy' else 10.,
        train_seen_rows=11542755, cold_rows=23915712,
        resources=dict(allocated_cpus=4, user_cpu_ceiling=4, library_threads=1,
                       job_id='job', allowed_cpus=[0, 1, 2, 3]),
        cpu_checks=[dict(allowed_cpus=[0, 1, 2, 3],
            processes=[dict(pid=100, threads_checked=3)]) for _ in range(21)],
        memory=[memory(stage) for stage in ['after_runtime', 'after_backend']
                + [f'update{i + 1}' for i in range(20)]],
        **{key: False for key in gate.FLAGS})
    if backend == 'cached_seen':
        shared = dict(rows=11542755, cache_bytes=11542755 * 128 * 4, cache_sha256='4' * 64,
            standardization_stats_sha256='1' * 64, seen_rows_sha256='2' * 64, source_sha256='3' * 64)
        report['cache'] = dict(shared, asset_rows=35458467, dimensions=128,
                              kind='fixed_standardized_input_features_not_projection')
        report['cache_verification'] = dict(shared, status='passed', feature_values_bitwise_equal=True)
    kwargs = dict(manifest=manifest, config=config, setting=setting, examples=examples(),
        manifest_sha256='d' * 64, fixture_entry=fixture_entry, job_id='job', expected_binding=binding)
    return report, kwargs


def report(size=1024, backend='lazy', negatives=8192):
    template, kwargs = report_template(size, backend, negatives)
    # Immutable prefix examples can be shared; copied bindings remain independent
    # so tampering a report cannot silently tamper the independent expected value.
    return copy.deepcopy(template), dict(kwargs, expected_binding=copy.deepcopy(kwargs['expected_binding']))


@pytest.mark.parametrize('size,backend,negatives', [
    (1024, 'lazy', 8192), (1024, 'cached_seen', 32768), (2048, 'cached_seen', 8192),
    (4096, 'cached_seen', 32768),
])
def test_valid_reports_keep_all_real_prefix_and_access_contracts(size, backend, negatives):
    r, kwargs = report(size, backend, negatives)
    gate.validate_report(r, **kwargs)


@pytest.mark.parametrize('change', ['history_calls', 'history_rows', 'raw_calls', 'raw_rows',
    'std_rows', 'failed', 'overhead', 'catalog_copy'])
def test_history_and_catalog_diagnostics_cannot_hide_or_double_count_reads(change):
    r, kwargs = report()
    row = r['steps'][0]
    access = row['access']
    if change == 'history_calls': access['history']['calls'] -= 1
    elif change == 'history_rows': access['history']['rows'] += 1
    elif change == 'raw_calls': access['raw']['calls'] = 1
    elif change == 'raw_rows': access['raw']['rows'] += 1
    elif change == 'std_rows': access['standardized']['rows'] -= 1
    elif change == 'failed': access['history']['failed_calls'] = 1
    elif change == 'overhead': access['standardization_overhead_wall_seconds'] = .5
    else: access['catalog'] = dict(row['catalog_fetch'], calls=2)
    with pytest.raises(AssertionError):
        gate.validate_access(row, kwargs['setting'], examples()[:1024])


def test_cached_training_must_not_touch_raw_or_standardizer():
    r, kwargs = report(1024, 'cached_seen')
    r['steps'][0]['access']['raw']['calls'] = 1
    with pytest.raises(AssertionError):
        gate.validate_access(r['steps'][0], kwargs['setting'], examples()[:1024])


def test_unavailable_io_is_explicit_not_zero_storage_claim():
    delta = io_delta()
    gate.validate_io(delta)
    delta['io']['read_bytes'] = None
    delta['unavailable_fields'] = ['read_bytes']
    gate.validate_io(delta)
    delta['unavailable_fields'] = []
    with pytest.raises(AssertionError): gate.validate_io(delta)


@pytest.mark.parametrize('change', ['negative', 'fault', 'pid', 'time', 'schema'])
def test_bad_io_deltas_are_rejected(change):
    delta = io_delta()
    if change == 'negative': delta['io']['read_bytes'] = -1
    elif change == 'fault': delta['major_faults'] = True
    elif change == 'pid': delta['pid'] = 0
    elif change == 'time': delta['cpu_seconds'] = float('nan')
    else: delta['schema'] = 'unknown'
    with pytest.raises(AssertionError): gate.validate_io(delta)


def test_cache_equivalence_ignores_backend_timing_not_learning():
    lazy, _ = report()
    cached, _ = report(1024, 'cached_seen')
    cached['steps'][0]['training_wall_seconds'] *= 9
    gate.require_exact_cache_equivalence(lazy, cached)


@pytest.mark.parametrize('change', ['batch', 'negative', 'initial', 'final', 'progress', 'loss', 'gpu'])
def test_equivalence_never_applies_across_batch_or_changed_learning(change):
    lazy, _ = report()
    cached, _ = report(1024, 'cached_seen')
    if change == 'batch': cached['batch_size'] = 2048
    elif change == 'negative': cached['num_negatives'] = 32768
    elif change == 'initial': cached['initial_state_sha256'] = 'other'
    elif change == 'final': cached['learning_state_sha256'] = 'other'
    elif change == 'progress': cached['final_progress']['updates'] = 19
    elif change == 'loss': cached['steps'][0]['training']['dense_loss'] += 1e-10
    else: cached['gpu']['uuid'] = 'another'
    with pytest.raises(AssertionError): gate.require_exact_cache_equivalence(lazy, cached)


@pytest.mark.parametrize('change', ['scope', 'binding', 'cpu', 'memory', 'cache_bytes', 'cache_hash',
    'cache_source', 'progress', 'identity', 'candidate', 'warmup', 'gpu_memory'])
def test_top_level_report_rejects_unsafe_or_misleading_artifacts(change):
    r, kwargs = report(1024, 'cached_seen')
    if change == 'scope': r['validation_test_records_read'] = True
    elif change == 'binding': r['checkpoint_binding']['probe_batch_size'] = 4096
    elif change == 'cpu': r['resources']['allocated_cpus'] = 24
    elif change == 'memory': r['memory'][1]['stage'] = 'wrong'
    elif change == 'cache_bytes': r['cache']['cache_bytes'] += 1
    elif change == 'cache_hash': r['cache_verification']['cache_sha256'] = '9' * 64
    elif change == 'cache_source': r['cache']['source_sha256'] = '9' * 64
    elif change == 'progress': r['final_progress']['updates'] = 19
    elif change == 'identity': r['steps'][0]['training']['prefix_identity_sha256'] = 'changed'
    elif change == 'candidate': r['steps'][0]['training']['dense_candidate_classes'] += 1
    elif change == 'warmup': r['steps'][4]['measured'] = True
    else: r['steps'][0]['peak_gpu_reserved_bytes'] = 25 * 1024**3
    with pytest.raises((AssertionError, ValueError)):
        gate.validate_report(r, **kwargs)


def oom_report():
    r, kwargs = report(2048, 'cached_seen')
    r.update(status='cuda_oom', failure_stage='training_attempt', failure_attempt=4,
             error='torch.cuda.OutOfMemoryError: synthetic')
    r.pop('learning_state_sha256')
    r['steps'] = r['steps'][:3]
    r['cpu_checks'] = r['cpu_checks'][:5]
    r['memory'] = r['memory'][:5] + [memory('failure')]
    return r, kwargs


def test_large_batch_training_oom_is_a_capacity_result():
    r, kwargs = oom_report()
    gate.validate_report(r, **kwargs)


@pytest.mark.parametrize('change', ['stage', 'error', 'attempt', 'baseline'])
def test_other_failures_are_not_capacity_results(change):
    r, kwargs = oom_report()
    if change == 'stage': r['failure_stage'] = 'cache_build'
    elif change == 'error': r['error'] = 'FileNotFoundError'
    elif change == 'attempt': r['failure_attempt'] = 5
    else:
        r['batch_size'] = 1024
    with pytest.raises(AssertionError): gate.validate_report(r, **kwargs)


def test_summary_excludes_five_warmups_and_reports_variability_and_tail():
    r, _ = report(2048, 'cached_seen')
    result = gate.summarize_case(r)
    assert result['measured_steps'] == 15 and result['warmup_steps'] == 5
    assert result['mean_step_seconds'] == result['median_step_seconds'] == 8.
    assert result['min_step_seconds'] == 1. and result['max_step_seconds'] == 15.
    assert result['step_seconds_sample_std'] == statistics.stdev(range(1, 16))
    assert result['step_seconds_cv'] == statistics.stdev(range(1, 16)) / 8
    assert result['prefix_examples_per_second'] == 256.
    assert result['prefix_batches_per_pass'] == math.ceil(45996113 / 2048)
    assert result['indicative_single_prefix_pass_training_hours'] == math.ceil(45996113 / 2048) * 8 / 3600
    assert result['one_time_setup_cache_seconds'] == 60.
    assert result['indicative_pass_plus_setup_cache_hours'] == (math.ceil(45996113 / 2048) * 8 + 60) / 3600
    assert 'not end-to-end' in result['timing_scope'] and 'do not sum' in result['access_scope']


@pytest.mark.parametrize('change', ['status', 'missing_step', 'nonfinite', 'warmup'])
def test_summary_cannot_hide_failed_or_missing_measurements(change):
    r, _ = report()
    if change == 'status': r['status'] = 'cuda_oom'
    elif change == 'missing_step': r['steps'].pop()
    elif change == 'nonfinite': r['steps'][8]['training_wall_seconds'] = float('nan')
    else: r['steps'][0]['measured'] = True
    with pytest.raises(AssertionError): gate.summarize_case(r)
