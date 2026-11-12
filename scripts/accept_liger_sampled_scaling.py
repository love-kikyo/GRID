"""Independent cache-equivalence, access-accounting and batch-capacity gate.

This reads bound small reports and the training fixture, never model tensors.
No throughput result grants a formal budget or establishes learning quality.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics

from scripts.accept_liger_efficiency import (
    expected_identity, validate_fixture, validate_memory_snapshot, validate_resource_report,
)
from scripts.accept_liger_sampled import _integer, _finite, _digest, FLAGS
from scripts.probe_liger_cpu4 import sha, write_new


ATTEMPTS = 20
WARMUP = 5
FIXTURE_PREFIXES = 81920


def validate_numpy_policy(policy, expected_version=None):
    assert policy['environment_value'] == '0'
    assert policy['effective_madvise_hugepage'] is False
    assert isinstance(policy['numpy_version'], str) and policy['numpy_version']
    assert policy['verification_api'] == 'numpy._core._multiarray_umath._get_madvise_hugepage'
    assert 'current NumPy process only' in policy['scope']
    assert 'no global kernel settings written' in policy['scope']
    assert 'not a proven cause' in policy['interpretation']
    if expected_version is not None:
        assert policy['numpy_version'] == expected_version


def validate_phase_progress(report):
    phases = report['phase_progress']
    if report['status'] == 'skipped_after_cuda_oom':
        assert phases == []
        return
    names = ['runtime_initialization', 'after_runtime']
    if report['backend'] == 'cached_seen':
        for stage in ('cache_build', 'cache_verification'):
            names.append(stage)
            chunks = list(range(16, math.ceil(11542755 / 65536), 16)) + [math.ceil(11542755 / 65536)]
            progress = [row for row in phases if row['phase'] == stage + '_progress']
            assert [row['completed_chunks'] for row in progress] == chunks
            assert all(row['total_rows'] == 11542755 and row['completed_rows'] ==
                       min(row['completed_chunks'] * 65536, 11542755) for row in progress)
            names.extend([stage + '_progress'] * len(chunks))
            names.append('after_' + stage)
    names.append('after_backend')
    if report['status'] == 'passed':
        names.extend(['training_complete', 'complete'])
    assert [row['phase'] for row in phases] == names
    previous = -1.
    pids = set()
    for row in phases:
        assert _finite(row['elapsed_seconds']) and row['elapsed_seconds'] >= previous
        previous = row['elapsed_seconds']
        assert row['allowed_cpus'] == report['resources']['allowed_cpus']
        validate_memory_snapshot(row['memory'])
        io = row['io']
        assert io['schema'] == 'liger.process_access_io.v1' and _integer(io['pid'], 1)
        pids.add(io['pid'])
        assert _finite(io['monotonic_seconds']) and _finite(io['process_cpu_seconds'])
        assert _integer(io['minor_faults']) and _integer(io['major_faults'])
        assert isinstance(io['errors'], list) and isinstance(io['timing_scope'], str)
        assert set(io['unavailable_fields']) == {key for key, value in io['io'].items() if value is None}
        assert all(value is None or _integer(value) for value in io['io'].values())
    assert len(pids) == 1


def _counter(value, calls, rows):
    assert value['calls'] == calls and value['rows'] == rows
    assert value['failed_calls'] == value['unknown_rows_calls'] == 0
    assert _finite(value['wall_seconds']) and _finite(value['cpu_seconds'])


def validate_access(row, setting, batch):
    """History and catalog accesses are different scopes; neither is GPU time."""
    access = row['access']
    history_rows = sum(len(e.sequence[:-1][-7:]) for e in batch)
    candidates = setting['num_negatives'] + len({e.sequence[-1] for e in batch})
    _counter(access['history'], len(batch), history_rows)
    if setting['backend'] == 'lazy':
        _counter(access['raw'], len(batch) + 1, history_rows + candidates)
        _counter(access['standardized'], len(batch) + 1, history_rows + candidates)
    else:
        assert setting['backend'] == 'cached_seen'
        _counter(access['raw'], 0, 0)
        _counter(access['standardized'], 0, 0)
    for clock in ('wall', 'cpu'):
        overhead = access[f'standardization_overhead_{clock}_seconds']
        assert _finite(overhead)
        delta = access['standardized'][f'{clock}_seconds'] - access['raw'][f'{clock}_seconds']
        assert math.isclose(overhead, delta, rel_tol=1e-9, abs_tol=1e-8)
    assert access['catalog'] == row['catalog_fetch']
    assert isinstance(access['timing_scope'], str) and access['timing_scope']


def validate_io(delta):
    """Validate diagnostic counters without treating zero bytes as no I/O."""
    assert isinstance(delta, dict)
    # The profiler owns the exact process-counter schema; validate values here
    # once, independently of the raw before/after snapshots.
    assert delta['schema'] == 'liger.process_access_io_delta.v1'
    assert _integer(delta['pid'], 1)
    assert isinstance(delta['errors'], list)
    assert isinstance(delta['unavailable_fields'], list)
    assert isinstance(delta['counter_reset_fields'], list)
    assert isinstance(delta['timing_scope'], str) and delta['timing_scope']
    for key in ('minor_faults', 'major_faults'):
        assert _integer(delta[key])
    for key in ('wall_seconds', 'cpu_seconds'):
        assert _finite(delta[key])
    expected_fields = {'rchar', 'wchar', 'syscr', 'syscw', 'read_bytes', 'write_bytes', 'cancelled_write_bytes'}
    assert set(delta['io']) == expected_fields
    for key, value in delta['io'].items():
        assert value is None or _integer(value)
    assert set(delta['unavailable_fields']) == {key for key, value in delta['io'].items() if value is None}
    assert set(delta['counter_reset_fields']) <= set(delta['unavailable_fields'])


def validate_training_steps(report, setting, examples, objective):
    assert len(examples) == FIXTURE_PREFIXES
    steps = report['steps']
    if report['status'] == 'passed':
        assert len(steps) == ATTEMPTS
    else:
        assert report['status'] == 'cuda_oom' and setting['batch_size'] > 1024
        assert _integer(report['failure_attempt'], 1) and report['failure_attempt'] <= ATTEMPTS
        assert len(steps) == report['failure_attempt'] - 1
    size = setting['batch_size']
    for index, row in enumerate(steps):
        attempt = index + 1
        step = row['training']
        batch = examples[index * size:attempt * size]
        forced = len({e.sequence[-1] for e in batch})
        assert row['measured'] is (index >= WARMUP)
        assert step['batch'] == step['attempt'] == step['updates'] == attempt
        assert step['rows'] == step['prefix_count'] == size
        assert step['skipped'] is False and step['gradient_tensors'] == 152
        assert step['dense_objective'] == objective and step['sampling_attempt'] == index
        assert step['forced_positive_classes'] == forced
        assert step['sampled_negative_classes'] == setting['num_negatives']
        assert step['dense_candidate_classes'] == forced + setting['num_negatives']
        assert step['prefix_identity_sha256'] == step['batch_identity_sha256'] == expected_identity(batch)
        assert step['final_target_count'] == sum(e.is_final for e in batch)
        assert step['batch_unique_parents'] == len({e.parent_user_id for e in batch})
        assert step['first_user'] == batch[0].parent_user_id and step['last_user'] == batch[-1].parent_user_id
        assert abs(step['learning_rate'] - .0003 * attempt / 963) < 1e-16
        assert all(_finite(step[key]) for key in ('sid_loss', 'dense_loss', 'total_loss', 'grad_norm'))
        assert abs(step['sid_loss'] + step['dense_loss'] - step['total_loss']) < 1e-5
        assert _finite(step['scale_before'], minimum=1e-30)
        assert _finite(step['scale_after'], minimum=step['scale_before'])
        assert _finite(row['training_wall_seconds'], minimum=1e-30)
        assert _integer(row['peak_gpu_allocated_bytes'], 1)
        assert _integer(row['peak_gpu_reserved_bytes'], row['peak_gpu_allocated_bytes'])
        fetch = row['catalog_fetch']
        assert fetch['calls'] == 1 and fetch['rows'] == forced + setting['num_negatives']
        assert fetch['failed_calls'] == fetch['unknown_rows_calls'] == 0
        assert _finite(fetch['cpu_fetch_wall_seconds'])
        validate_access(row, setting, batch)
        validate_io(row['io_delta'])


def validate_cache(report, expected_binding):
    if report['backend'] == 'lazy':
        assert report['cache_build_seconds'] == report['cache_verify_seconds'] == 0.
        assert 'cache' not in report and 'cache_verification' not in report
        return
    assert report['backend'] == 'cached_seen'
    assert _finite(report['cache_build_seconds'], minimum=1e-30)
    assert _finite(report['cache_verify_seconds'], minimum=1e-30)
    cache, verified = report['cache'], report['cache_verification']
    assert verified['status'] == 'passed' and verified['feature_values_bitwise_equal'] is True
    assert cache['rows'] == verified['rows'] == 11542755
    assert cache['cache_bytes'] == verified['cache_bytes'] == 11542755 * 128 * 4
    assert _digest(cache['cache_sha256']) and cache['cache_sha256'] == verified['cache_sha256']
    assert cache['asset_rows'] == 35458467 and cache['dimensions'] == 128
    assert cache['kind'] == 'fixed_standardized_input_features_not_projection'
    expected = dict(standardization_stats_sha256=expected_binding['standardization_stats_sha256'],
        seen_rows_sha256=expected_binding['standardization_seen_rows_sha256'],
        source_sha256=expected_binding['assets']['model_embedding_copy'])
    for key, value in expected.items():
        assert cache[key] == verified[key] == value


def validate_report(report, *, manifest, config, setting, examples,
                    manifest_sha256, fixture_entry, job_id, expected_binding):
    assert report['status'] in ('passed', 'cuda_oom', 'skipped_after_cuda_oom')
    assert report['manifest_sha256'] == manifest_sha256
    assert report['source_git_commit'] == manifest['source_git_commit']
    assert report['phase'] == manifest['phase'] and report['job_id'] == job_id
    assert all(report[key] == value for key, value in setting.items())
    assert all(report[key] is False for key in FLAGS)
    assert report['fixture_receipt'] == fixture_entry and _finite(report['elapsed_seconds'])
    validate_numpy_policy(report['numpy_memory_policy'])
    if report['status'] == 'skipped_after_cuda_oom':
        assert setting['batch_size'] == 4096 and setting['backend'] == 'cached_seen'
        assert report['steps'] == [] and report['memory'] == []
        assert isinstance(report['reason'], str) and report['reason']
        assert 'learning_state_sha256' not in report
        validate_phase_progress(report)
        return
    assert report['checkpoint_binding'] == expected_binding
    assert expected_binding['probe_case'] == setting['name']
    assert expected_binding['probe_batch_size'] == setting['batch_size']
    assert expected_binding['feature_backend'] == setting['backend']
    assert expected_binding['probe_fixture_sha256'] == fixture_entry['sha256']
    assert _digest(report['initial_state_sha256'])
    assert '4090' in report['gpu']['name'] and report['gpu']['uuid']
    assert _integer(report['gpu']['total_bytes'], 1)
    assert report['versions'] and isinstance(report['versions'], dict)
    validate_numpy_policy(report['numpy_memory_policy'], report['versions']['numpy'])
    assert _finite(report['setup_seconds'])
    assert report['train_seen_rows'] == 11542755 and report['cold_rows'] == 23915712
    validate_cache(report, expected_binding)
    validate_training_steps(report, setting, examples, expected_binding['dense_objective'])
    count = len(report['steps'])
    failure = report['status'] == 'cuda_oom'
    validate_resource_report(report, expected_checks=1 + count + int(failure))
    validate_phase_progress(report)
    stages = ['after_runtime', 'after_backend'] + [f'update{i + 1}' for i in range(count)]
    if failure:
        stages.append('failure')
    assert [row['stage'] for row in report['memory']] == stages
    for row in report['memory']:
        validate_memory_snapshot(row)
    assert all(row['peak_gpu_reserved_bytes'] <= report['gpu']['total_bytes'] for row in report['steps'])
    if failure:
        assert setting['batch_size'] > 1024 and report['failure_stage'] == 'training_attempt'
        assert 'OutOfMemoryError' in report['error'] and 'learning_state_sha256' not in report
        return
    from src.baselines.liger_prefix_training import check_progress
    budget = {key: config['training'][key] for key in ('max_optimizer_steps', 'max_attempts')}
    budget.update(batch_size=setting['batch_size'], prefixes_per_epoch=manifest['counts']['prefix_examples_per_pass'])
    p = report['final_progress']
    check_progress(p, budget)
    assert p['updates'] == p['attempts'] == p['next_batch'] == ATTEMPTS
    assert p['skipped_updates'] == 0 and p['epoch'] == 0
    assert p['prefix_count'] == p['samples_in_epoch'] == ATTEMPTS * setting['batch_size']
    assert p['final_count'] == sum(e.is_final for e in examples[:ATTEMPTS * setting['batch_size']])
    assert p['dense_objective'] == expected_binding['dense_objective']
    assert p['best'] is None and p['validations'] == []
    assert _digest(report['learning_state_sha256'])


def require_exact_cache_equivalence(lazy, cached):
    assert lazy['status'] == cached['status'] == 'passed'
    assert lazy['backend'] == 'lazy' and cached['backend'] == 'cached_seen'
    assert lazy['batch_size'] == cached['batch_size'] == 1024
    assert lazy['num_negatives'] == cached['num_negatives']
    for key in ('gpu', 'versions', 'initial_state_sha256', 'learning_state_sha256',
                'final_progress', 'fixture_receipt', 'numpy_memory_policy'):
        assert lazy[key] == cached[key], f'Cache equivalence mismatch: {key}'
    assert [row['training'] for row in lazy['steps']] == [row['training'] for row in cached['steps']]


def summarize_case(report, prefix_count=45996113):
    assert report['status'] == 'passed' and len(report['steps']) == ATTEMPTS
    measured = [row for row in report['steps'] if row['measured']]
    assert len(measured) == ATTEMPTS - WARMUP == 15
    seconds = [row['training_wall_seconds'] for row in measured]
    assert all(_finite(value, minimum=1e-30) for value in seconds)
    mean, std = statistics.mean(seconds), statistics.stdev(seconds)
    batches = math.ceil(prefix_count / report['batch_size'])
    setup, build, verify = [report[key] for key in ('setup_seconds', 'cache_build_seconds', 'cache_verify_seconds')]
    return dict(name=report['name'], backend=report['backend'], num_negatives=report['num_negatives'],
        batch_size=report['batch_size'], measured_steps=15, warmup_steps=5,
        mean_step_seconds=mean, median_step_seconds=statistics.median(seconds),
        min_step_seconds=min(seconds), max_step_seconds=max(seconds),
        step_seconds_sample_std=std, step_seconds_cv=std / mean,
        prefix_examples_per_second=report['batch_size'] / mean,
        max_gpu_allocated_bytes=max(row['peak_gpu_allocated_bytes'] for row in report['steps']),
        max_gpu_reserved_bytes=max(row['peak_gpu_reserved_bytes'] for row in report['steps']),
        max_process_self_hwm_bytes=max(row['high_water_rss_bytes'] for row in report['memory']),
        setup_seconds=setup, cache_build_seconds=build, cache_verify_seconds=verify,
        one_time_setup_cache_seconds=setup + build + verify,
        mean_catalog_fetch_seconds=statistics.mean(row['catalog_fetch']['cpu_fetch_wall_seconds'] for row in measured),
        mean_history_access_seconds=statistics.mean(row['access']['history']['wall_seconds'] for row in measured),
        mean_raw_gather_seconds=statistics.mean(row['access']['raw']['wall_seconds'] for row in measured),
        mean_standardized_access_seconds=statistics.mean(row['access']['standardized']['wall_seconds'] for row in measured),
        mean_nested_standardization_overhead_seconds=statistics.mean(
            row['access']['standardization_overhead_wall_seconds'] for row in measured),
        prefix_examples_per_pass=prefix_count, prefix_batches_per_pass=batches,
        indicative_single_prefix_pass_training_hours=batches * mean / 3600,
        indicative_pass_plus_setup_cache_hours=(batches * mean + setup + build + verify) / 3600,
        timing_scope='fifteen synchronized full training steps after five warmup steps; '
            'preloaded real prefix inputs; excludes input streaming, setup, cache build/verification and validation; '
            'mean applied also to the tail; not end-to-end or convergence time',
        access_scope='nested CPU access scopes overlap each other and may overlap queued GPU work; '
            'do not sum them or subtract them from train wall to claim pure GPU time',
        memory_scope='PyTorch CUDA allocated/reserved peaks over all twenty attempts; CPU process-self lifetime '
            'RSS high-water, not whole-job/cgroup RAM')


def _expected_binding(m, setting, fixture_entry):
    from scripts.accept_liger_sampled import _expected_binding as prior_binding
    return dict(prior_binding(m, setting, fixture_entry), feature_backend=setting['backend'],
                probe_batch_size=setting['batch_size'])


def load_case(path, m, c, setting, examples, fixture_entry):
    target = path.parent / f"{setting['name']}.json"
    report = json.loads(target.read_text())
    validate_report(report, manifest=m, config=c, setting=setting, examples=examples,
        manifest_sha256=sha(path), fixture_entry=fixture_entry, job_id=os.environ['SLURM_JOB_ID'],
        expected_binding=_expected_binding(m, setting, fixture_entry))
    return report, dict(path=str(target), sha256=sha(target))


def _context(path):
    from scripts.liger_sampled_scaling_manifest import binding
    from scripts.probe_liger_sampled_scaling import load_fixture
    m, c = binding(path)
    fixture, examples, entry = load_fixture(path, m)
    validate_fixture(fixture, examples, asset_rows=m['asset_rows'], expected_prefixes=FIXTURE_PREFIXES, batch_size=1024)
    validate_numpy_policy(fixture['numpy_memory_policy'])
    return m, c, fixture, examples, entry


def equivalence(path):
    from scripts.probe_liger_sampled_scaling import deadline
    m, c, fixture, examples, fixture_entry = _context(path)
    reports, entries = {}, []
    for setting in c['cases'][:4]:
        report, entry = load_case(path, m, c, setting, examples, fixture_entry)
        reports[setting['name']] = report
        entries.append(entry)
    for prefix in ('8k', '32k'):
        require_exact_cache_equivalence(reports[f'lazy{prefix}1024'], reports[f'cached{prefix}1024'])
    deadline(fixture)
    write_new(path.parent / 'cache_equivalence.json', dict(status='passed_exact_cache_equivalence',
        job_id=os.environ['SLURM_JOB_ID'], manifest_sha256=sha(path), source_git_commit=m['source_git_commit'],
        reports=entries, fixture_receipt=fixture_entry, compared='weights, optimizer, scheduler, scaler, RNG, '
            'progress and training mode; separately validated execution binding excluded from state digest',
        numpy_memory_policy=reports['lazy8k1024']['numpy_memory_policy'],
        **{key: False for key in FLAGS}))
    print('[liger-scaling] both K cache equivalence checks passed; larger batches allowed', flush=True)


def check_equivalence_receipt(path):
    target = path.parent / 'cache_equivalence.json'
    r = json.loads(target.read_text())
    assert r['status'] == 'passed_exact_cache_equivalence'
    assert r['manifest_sha256'] == sha(path) and r['job_id'] == os.environ['SLURM_JOB_ID']
    assert all(r[key] is False for key in FLAGS)
    validate_numpy_policy(r['numpy_memory_policy'])
    for entry in r['reports']:
        assert sha(entry['path']) == entry['sha256']
    return dict(path=str(target), sha256=sha(target))


def accept(path):
    from scripts.probe_liger_sampled_scaling import deadline
    m, c, fixture, examples, fixture_entry = _context(path)
    equivalence_entry = check_equivalence_receipt(path)
    eq = json.loads(Path(equivalence_entry['path']).read_text())
    assert eq['source_git_commit'] == m['source_git_commit'] and eq['fixture_receipt'] == fixture_entry
    reports, entries = {}, [fixture_entry, equivalence_entry]
    for setting in c['cases']:
        r, entry = load_case(path, m, c, setting, examples, fixture_entry)
        if setting['batch_size'] > 1024:
            assert r['equivalence_receipt'] == equivalence_entry
        reports[setting['name']] = r
        entries.append(entry)
    assert eq['reports'] == entries[2:6]
    for prefix in ('8k', '32k'):
        require_exact_cache_equivalence(reports[f'lazy{prefix}1024'], reports[f'cached{prefix}1024'])
        previous, larger = reports[f'cached{prefix}2048'], reports[f'cached{prefix}4096']
        assert (larger['status'] == 'skipped_after_cuda_oom') == (previous['status'] == 'cuda_oom')
    initialized = [r for r in reports.values() if r['status'] != 'skipped_after_cuda_oom']
    reference = initialized[0]
    for r in initialized[1:]:
        for key in ('gpu', 'versions', 'initial_state_sha256', 'fixture_receipt', 'numpy_memory_policy'):
            assert r[key] == reference[key]
    cached = [r for r in initialized if r['backend'] == 'cached_seen']
    assert len({r['cache']['cache_sha256'] for r in cached}) == 1
    passed = all(r['status'] == 'passed' for r in reports.values())
    summaries = [summarize_case(r, m['counts']['prefix_examples_per_pass'])
                 for r in reports.values() if r['status'] == 'passed']
    deadline(fixture)
    write_new(path.parent / 'scaling_acceptance.json', dict(
        status='passed_resource_probe' if passed else 'capacity_limited_resource_probe',
        job_id=os.environ['SLURM_JOB_ID'], source_git_commit=m['source_git_commit'], manifest_sha256=sha(path),
        reports=entries, gpu=reference['gpu'], summaries=summaries,
        numpy_memory_policy=reference['numpy_memory_policy'],
        capacity={name: r['status'] for name, r in reports.items()},
        exact_cache_equivalence_both_negative_counts=True, data_preparation=fixture['data_batches'],
        cpu_ceiling=4, library_threads=1, formal_budget_approved=False, formal_batch_selected=False,
        **{key: False for key in FLAGS},
        limits='backend order is counterbalanced across K but filesystem cache/shared load are not controlled; '
            'cross-batch supervised grouping differs; resource measurements do not establish learning equivalence',
        next_action='author reviews stable throughput, memory and validation/learning costs before formal budget'))
    print('[liger-scaling] resource and access gate accepted; no automatic formal training', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['equivalence', 'accept'])
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    (equivalence if args.mode == 'equivalence' else accept)(args.manifest.resolve())
