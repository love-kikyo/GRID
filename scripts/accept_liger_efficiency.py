"""Independent small-artifact checks; never promotes a probe to formal training."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import statistics

from scripts.liger_efficiency_manifest import binding, RECEIPT
from scripts.probe_liger_cpu4 import sha, write_new


def _integer(value, *, minimum=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _finite(value, *, minimum=0.):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= minimum)


def validate_memory_snapshot(snapshot):
    """Require usable process-self counters, not a claim about whole-job RAM."""
    assert snapshot['schema'] == 'liger.process_self_resources.v1'
    assert snapshot['errors'] == []
    for key in ('rss_bytes', 'high_water_rss_bytes', 'virtual_memory_bytes',
                'anonymous_rss_bytes', 'file_rss_bytes', 'shared_rss_bytes',
                'swap_bytes', 'ru_maxrss_bytes'):
        assert _integer(snapshot[key]), f'Invalid memory counter: {key}'
    assert snapshot['rss_bytes'] > 0
    assert snapshot['high_water_rss_bytes'] >= snapshot['rss_bytes']
    assert snapshot['virtual_memory_bytes'] >= snapshot['rss_bytes']
    assert snapshot['ru_maxrss_bytes'] > 0
    # Process RSS (including shared mappings) is not Slurm/cgroup job memory.
    # Keep the 24 GiB job allocation in the launcher, not a false RSS oracle.
    assert _integer(snapshot['threads'], minimum=1)
    assert _integer(snapshot['clock_ticks_per_second'], minimum=1)
    for key in ('monotonic_seconds', 'proc_cpu_user_seconds', 'proc_cpu_system_seconds',
                'proc_cpu_total_seconds', 'resource_cpu_user_seconds', 'resource_cpu_system_seconds'):
        assert _finite(snapshot[key]), f'Invalid process counter: {key}'
    assert math.isclose(snapshot['proc_cpu_total_seconds'],
                        snapshot['proc_cpu_user_seconds'] + snapshot['proc_cpu_system_seconds'],
                        rel_tol=1e-12, abs_tol=1e-9)


def validate_resource_report(report, *, expected_checks):
    resources = report['resources']
    assert resources['allocated_cpus'] == resources['user_cpu_ceiling'] == 4
    assert resources['library_threads'] == 1 and resources['job_id'] == report['job_id']
    allowed = resources['allowed_cpus']
    assert isinstance(allowed, list) and 1 <= len(allowed) <= 4
    assert all(_integer(cpu) for cpu in allowed) and sorted(set(allowed)) == allowed
    assert len(report['cpu_checks']) == expected_checks
    for check in report['cpu_checks']:
        assert check['allowed_cpus'] == allowed and check['processes']
        for process in check['processes']:
            assert _integer(process['pid'], minimum=1)
            assert _integer(process['threads_checked'], minimum=1)


def validate_fixture(receipt, examples, *, asset_rows, expected_prefixes=16384, batch_size=1024):
    """Recompute bounded-prefix structure/counts without rereading TFRecords."""
    assert len(examples) == receipt['prefix_examples'] == expected_prefixes
    assert expected_prefixes % batch_size == 0
    identities, parents, longest = set(), {}, {}
    for e in examples:
        assert _integer(e.parent_user_id, minimum=-(2**63)) and e.parent_user_id < 2**63
        assert re.fullmatch('[0-9a-fA-F]{64}', e.parent_serialized_sha256)
        assert _integer(e.parent_sequence_length, minimum=2) and e.parent_sequence_length <= 1000
        start = max(0, e.parent_sequence_length - 8)
        count = e.parent_sequence_length - start - 1
        assert _integer(e.window_start_index) and e.window_start_index == start
        assert _integer(e.parent_prefix_count, minimum=1) and e.parent_prefix_count == count
        assert _integer(e.prefix_target_index, minimum=1) and e.prefix_target_index <= count
        assert len(e.sequence) == e.prefix_target_index + 1
        assert all(_integer(row) and row < asset_rows for row in e.sequence)
        assert isinstance(e.is_final, bool) and e.is_final == (e.prefix_target_index == count)
        identity = (e.parent_user_id, e.prefix_target_index)
        assert identity not in identities, 'Duplicate fixture prefix'
        identities.add(identity)
        parent = (e.parent_serialized_sha256, e.parent_sequence_length, start, count)
        assert parents.setdefault(e.parent_user_id, parent) == parent
        previous = longest.get(e.parent_user_id, ())
        common = min(len(previous), len(e.sequence))
        assert tuple(previous[:common]) == tuple(e.sequence[:common])
        if len(e.sequence) > len(previous):
            longest[e.parent_user_id] = e.sequence
    assert receipt['unique_parents'] == len(parents)
    assert receipt['final_target_count'] == sum(e.is_final for e in examples)
    batches = receipt['data_batches']
    assert len(batches) == expected_prefixes // batch_size
    assert [row['index'] for row in batches] == list(range(len(batches)))
    assert all(row['rows'] == batch_size and _finite(row['seconds']) for row in batches)
    assert _finite(receipt['started_unix_seconds']) and _finite(receipt['deadline_unix_seconds'])
    assert receipt['deadline_unix_seconds'] - receipt['started_unix_seconds'] == 2400.
    assert _finite(receipt['elapsed_seconds']) and receipt['elapsed_seconds'] < 2400.
    assert not any(receipt[k] for k in ('formal_training_ready', 'automatic_successor',
        'validation_test_records_read', 'formal_checkpoint_produced'))
    validate_resource_report(receipt, expected_checks=2)
    validate_memory_snapshot(receipt['memory'])


def expected_identity(examples):
    import numpy as np
    from src.baselines.liger_state import tree_digest
    fields = {
        'parent_user_ids': np.asarray([e.parent_user_id for e in examples], dtype=np.int64),
        'parent_serialized_sha256': [e.parent_serialized_sha256 for e in examples],
        'parent_sequence_lengths': np.asarray([e.parent_sequence_length for e in examples], dtype=np.int64),
        'window_start_indices': np.asarray([e.window_start_index for e in examples], dtype=np.int64),
        'prefix_target_indices': np.asarray([e.prefix_target_index for e in examples], dtype=np.int64),
        'parent_target_indices': np.asarray([e.parent_target_index for e in examples], dtype=np.int64),
        'parent_prefix_counts': np.asarray([e.parent_prefix_count for e in examples], dtype=np.int64),
        'is_final': np.asarray([e.is_final for e in examples], dtype=np.bool_),
        'target_rows': np.asarray([e.sequence[-1] for e in examples], dtype=np.int64),
        'history_rows': [list(e.sequence[:-1][-7:]) for e in examples],
    }
    return tree_digest(fields)


def load_case(path, m, setting, examples):
    target = path.parent / f"{setting['name']}.json"
    r = json.loads(target.read_text())
    assert r['manifest_sha256'] == sha(path) and r['source_git_commit'] == m['source_git_commit']
    assert r['job_id'] == os.environ['SLURM_JOB_ID'] and r['phase'] == m['phase']
    assert all(r[key] == value for key, value in setting.items())
    assert not any(r[k] for k in ('formal_training_ready', 'automatic_successor',
        'validation_test_records_read', 'formal_checkpoint_produced'))
    assert sha(r['fixture_receipt']['path']) == r['fixture_receipt']['sha256']
    if r['status'] in ('cuda_oom', 'skipped_after_cuda_oom'):
        assert setting['batch_size'] > 1024
        if r['status'] == 'cuda_oom':
            assert 'OutOfMemoryError' in r['error'] and 1 <= r['failure_attempt'] <= 4
            assert r['failure_stage'] == 'training_attempt'
            assert r['failure_attempt'] == len(r['steps']) + 1
            validate_resource_report(r, expected_checks=len(r['steps']) + 2)
            assert [row['stage'] for row in r['memory']] == (
                ['before_cache', 'after_cache']
                + [f'update{i + 1}' for i in range(len(r['steps']))] + ['failure'])
            for snapshot in r['memory']:
                validate_memory_snapshot(snapshot)
        return r, dict(path=str(target), sha256=sha(target))
    assert r['status'] == 'passed'
    assert '4090' in r['gpu']['name']
    validate_resource_report(r, expected_checks=5)
    assert [row['stage'] for row in r['memory']] == ['before_cache', 'after_cache', 'update1', 'update2', 'update3', 'update4']
    for snapshot in r['memory']:
        validate_memory_snapshot(snapshot)
    assert all(_finite(r[k]) for k in ('setup_seconds', 'cache_build_seconds', 'cache_verify_seconds'))
    assert r['train_seen_rows'] == 11542755 and r['cold_rows'] == 23915712
    assert len(r['steps']) == 4 and r['final_progress']['updates'] == r['final_progress']['attempts'] == 4
    assert r['final_progress']['skipped_updates'] == 0
    total_final = 0
    for i, row in enumerate(r['steps']):
        step = row['training']
        size = setting['batch_size']
        batch = examples[i * size:(i + 1) * size]
        assert row['measured'] == (i > 0)
        assert step['attempt'] == step['updates'] == step['batch'] == i + 1
        assert step['rows'] == step['prefix_count'] == size and not step['skipped']
        assert step['gradient_tensors'] == 152
        assert step['prefix_identity_sha256'] == step['batch_identity_sha256'] == expected_identity(batch)
        assert step['final_target_count'] == sum(e.is_final for e in batch)
        assert step['batch_unique_parents'] == len({e.parent_user_id for e in batch})
        assert step['first_user'] == batch[0].parent_user_id and step['last_user'] == batch[-1].parent_user_id
        total_final += step['final_target_count']
        assert abs(step['learning_rate'] - .0003 * (i + 1) / 963) < 1e-16
        assert all(math.isfinite(step[k]) for k in ('sid_loss', 'dense_loss', 'total_loss', 'grad_norm'))
        assert abs(step['sid_loss'] + step['dense_loss'] - step['total_loss']) < 4e-6
        assert math.isfinite(row['training_wall_seconds']) and row['training_wall_seconds'] > 0
        assert row['peak_gpu_allocated_bytes'] > 0 and row['peak_gpu_reserved_bytes'] >= row['peak_gpu_allocated_bytes']
        fetch = row['catalog_fetch']
        assert fetch['calls'] == 2 * math.ceil(11542755 / 65536) and fetch['rows'] == 2 * 11542755
        assert fetch['failed_calls'] == fetch['unknown_rows_calls'] == 0
        assert math.isfinite(fetch['cpu_fetch_wall_seconds']) and fetch['cpu_fetch_wall_seconds'] >= 0
        forward = row['model_forward']
        assert forward['calls'] == 1 and forward['failed_calls'] == 0 and forward['synchronized']
        assert forward['model_forward_wall_seconds'] > 0
    assert r['final_progress']['prefix_count'] == 4 * setting['batch_size']
    assert r['final_progress']['final_count'] == total_final
    if setting['backend'] == 'cached_seen':
        assert r['mixed_global_row_lookup_exact'] and r['cache_build_seconds'] > 0 and r['cache_verify_seconds'] > 0
        cache, verification = r['cache'], r['cache_verification']
        assert verification['status'] == 'passed' and verification['feature_values_bitwise_equal']
        assert cache['rows'] == verification['rows'] == 11542755
        assert cache['cache_bytes'] == verification['cache_bytes'] == 11542755 * 128 * 4
        assert cache['cache_sha256'] == verification['cache_sha256']
        assert cache['asset_rows'] == 35458467 and cache['dimensions'] == 128
        assert cache['kind'] == 'fixed_standardized_input_features_not_projection'
        stats = json.loads(Path(m['standardization']['path']).read_text())
        for key, expected in dict(standardization_stats_sha256=stats['stats_sha256'],
                seen_rows_sha256=stats['seen_rows_sha256'], source_sha256=stats['content_sha256']).items():
            assert cache[key] == verification[key] == expected
    return r, dict(path=str(target), sha256=sha(target))


def require_exact_equivalence(a, b):
    assert a['status'] == b['status'] == 'passed'
    assert a['backend'] == 'lazy' and b['backend'] == 'cached_seen'
    assert a['batch_size'] == b['batch_size'] == 1024
    assert a['gpu'] == b['gpu'] and a['initial_state_sha256'] == b['initial_state_sha256']
    assert a['fixture_receipt'] == b['fixture_receipt']
    assert [x['training'] for x in a['steps']] == [x['training'] for x in b['steps']]
    assert a['final_progress'] == b['final_progress']
    assert a['learning_state_sha256'] == b['learning_state_sha256']


def equivalence(path):
    from scripts.probe_liger_efficiency import load_fixture, deadline
    m, c = binding(path)
    fixture, examples, _ = load_fixture(path, m)
    a, a_entry = load_case(path, m, c['cases'][0], examples)
    b, b_entry = load_case(path, m, c['cases'][1], examples)
    require_exact_equivalence(a, b)
    # Frozen-prefix input order also agrees with the accepted v3 integration.
    prior = json.loads(RECEIPT.read_text())
    entry = prior['reports'][0]
    assert sha(entry['path']) == entry['sha256']
    old = json.loads(Path(entry['path']).read_text())
    assert [x['training']['prefix_identity_sha256'] for x in a['steps'][:3]] == [x['prefix_identity_sha256'] for x in old['steps']]
    deadline(fixture)
    write_new(path.parent / 'cache_equivalence.json', dict(status='passed_exact',
        job_id=os.environ['SLURM_JOB_ID'], manifest_sha256=sha(path), reports=[a_entry, b_entry],
        learning_state_sha256=a['learning_state_sha256'],
        compared='model, optimizer, scheduler, scaler, RNG, progress, model training mode; binding excluded',
        cache_values_exact=True, formal_training_ready=False, automatic_successor=False))
    print('[liger-efficiency] cache values, losses and complete learning state exactly match', flush=True)


def check_equivalence_receipt(path):
    target = path.parent / 'cache_equivalence.json'
    receipt = json.loads(target.read_text())
    assert receipt['status'] == 'passed_exact' and receipt['manifest_sha256'] == sha(path)
    assert receipt['job_id'] == os.environ['SLURM_JOB_ID']
    assert not receipt['formal_training_ready'] and not receipt['automatic_successor']
    for entry in receipt['reports']:
        assert sha(entry['path']) == entry['sha256']
    return dict(path=str(target), sha256=sha(target))


def summarize_case(report, prefix_count):
    measured = [x for x in report['steps'] if x['measured']]
    assert len(measured) == 3
    times = [x['training_wall_seconds'] for x in measured]
    mean = statistics.mean(times)
    batches = math.ceil(prefix_count / report['batch_size'])
    build, verify, setup = (report[k] for k in ('cache_build_seconds', 'cache_verify_seconds', 'setup_seconds'))
    assert all(_finite(value) for value in (build, verify, setup))
    compute = batches * mean
    return dict(name=report['name'], batch_size=report['batch_size'], measured_steps=3,
        mean_training_wall_seconds=mean, min_training_wall_seconds=min(times), max_training_wall_seconds=max(times),
        prefix_examples_per_second=report['batch_size'] / mean,
        mean_catalog_cpu_fetch_seconds=statistics.mean(x['catalog_fetch']['cpu_fetch_wall_seconds'] for x in measured),
        mean_model_forward_wall_seconds=statistics.mean(x['model_forward']['model_forward_wall_seconds'] for x in measured),
        setup_seconds=setup, cache_build_seconds=build, cache_verify_seconds=verify,
        one_time_setup_and_cache_seconds=setup + build + verify,
        prefix_examples_per_pass=prefix_count, batches_per_pass=batches,
        indicative_pass_compute_hours=compute / 3600,
        indicative_pass_compute_plus_cache_build_hours=(compute + build) / 3600,
        indicative_pass_compute_plus_setup_and_cache_hours=(compute + setup + build + verify) / 3600,
        max_gpu_allocated_bytes=max(x['peak_gpu_allocated_bytes'] for x in report['steps']),
        max_process_self_hwm_bytes=max(x['high_water_rss_bytes'] for x in report['memory']),
        memory_scope='process self lifetime RSS high-water; not whole-job/cgroup memory or GPU allocation',
        scope='three warm preloaded-input steps; mean used for every batch including tail; '
              'excludes streaming/validation/checkpoint/replay and warmup excess; '
              'setup/build/verification listed separately and included only in the named estimate; '
              'instrumented compute extrapolation, not end-to-end or convergence time')


def accept(path):
    from scripts.probe_liger_efficiency import load_fixture, deadline
    m, c = binding(path)
    fixture, examples, fixture_entry = load_fixture(path, m)
    equivalence_entry = check_equivalence_receipt(path)
    cases, artifacts = [], [fixture_entry, equivalence_entry]
    for setting in c['cases']:
        report, entry = load_case(path, m, setting, examples)
        cases.append(report)
        artifacts.append(entry)
    require_exact_equivalence(*cases[:2])
    if cases[3]['status'] == 'skipped_after_cuda_oom':
        assert cases[2]['status'] == 'cuda_oom'
    for r in cases[1:]:
        if r['status'] == 'passed':
            assert r['gpu'] == cases[0]['gpu'] and r['initial_state_sha256'] == cases[0]['initial_state_sha256']
    summaries = [summarize_case(r, m['counts']['prefix_examples_per_pass']) for r in cases if r['status'] == 'passed']
    reference = summaries[0]['prefix_examples_per_second']
    for summary in summaries:
        summary['throughput_ratio_to_lazy1024'] = summary['prefix_examples_per_second'] / reference
    deadline(fixture)
    write_new(path.parent / 'efficiency_acceptance.json', dict(status='passed_efficiency_probe',
        job_id=os.environ['SLURM_JOB_ID'], source_git_commit=m['source_git_commit'],
        manifest_sha256=sha(path), reports=artifacts, gpu=cases[0]['gpu'],
        exact_lazy_cached1024_equivalence=True, summaries=summaries,
        capacity={r['name']: r['status'] for r in cases},
        data_preparation=fixture['data_batches'],
        cpu_ceiling=4, library_threads=1, formal_training_ready=False,
        automatic_successor=False, validation_test_records_read=False,
        formal_batch_selected=False, formal_budget_approved=False,
        next_action='author review of practical cost; may stop LIGER, no automatic training'))
    print('[liger-efficiency] probe accepted; costs are indicative and require author review', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['equivalence', 'accept'])
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    (equivalence if args.mode == 'equivalence' else accept)(args.manifest.resolve())
