"""Independent acceptance of a bounded sampled-loss resource probe.

Only small, source-bound reports and the frozen training-prefix fixture are
read. A successful receipt is never a learning-quality or training-budget
approval. The two negative counts share data and initialization, not losses.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import statistics

from scripts.accept_liger_efficiency import (
    expected_identity, validate_fixture, validate_memory_snapshot,
    validate_resource_report,
)
from scripts.probe_liger_cpu4 import sha, write_new


ATTEMPTS = 12
WARMUP = 2
FLAGS = ('formal_training_ready', 'automatic_successor',
         'validation_test_records_read', 'formal_checkpoint_produced')


def _integer(value, minimum=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _finite(value, minimum=0.):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= minimum)


def _digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_training_steps(report, setting, examples, objective):
    """Recompute training identities and sampled-row counts from the fixture."""
    first = WARMUP + 1 if report['resumed'] else 1
    steps = report['steps']
    assert len(examples) == ATTEMPTS * setting['batch_size']
    if report['status'] == 'passed':
        assert len(steps) == ATTEMPTS - first + 1
    else:
        assert report['status'] == 'cuda_oom' and not report['resumed']
        assert _integer(report['failure_attempt'], 1) and report['failure_attempt'] <= ATTEMPTS
        assert len(steps) == report['failure_attempt'] - 1
    size = setting['batch_size']
    for attempt, row in enumerate(steps, first):
        step = row['training']
        batch = examples[(attempt - 1) * size:attempt * size]
        forced = len({e.sequence[-1] for e in batch})
        assert row['measured'] is (attempt > WARMUP)
        assert step['attempt'] == step['updates'] == step['batch'] == attempt
        assert step['rows'] == step['prefix_count'] == size == 1024
        assert step['skipped'] is False and step['gradient_tensors'] == 152
        assert step['dense_objective'] == objective
        assert step['sampling_attempt'] == attempt - 1
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


def validate_report(report, *, manifest, config, setting, examples,
                    manifest_sha256, fixture_entry, job_id, expected_binding=None):
    """Validate a report without loading a model or any checkpoint tensors."""
    assert report['status'] in ('passed', 'cuda_oom', 'skipped_after_cuda_oom')
    assert report['manifest_sha256'] == manifest_sha256
    assert report['source_git_commit'] == manifest['source_git_commit']
    assert report['phase'] == manifest['phase'] and report['job_id'] == job_id
    assert all(report[key] == value for key, value in setting.items())
    assert all(report[key] is False for key in FLAGS)
    assert isinstance(report['resumed'], bool)
    assert report['fixture_receipt'] == fixture_entry
    assert _finite(report['elapsed_seconds'])
    if report['status'] == 'skipped_after_cuda_oom':
        assert report['steps'] == [] and report['memory'] == []
        assert isinstance(report['reason'], str) and report['reason']
        assert 'final_state_sha256' not in report
        return

    from src.baselines.liger_sampled_catalog import sampling_binding
    objective = sampling_binding({key: setting[key] for key in ('num_negatives', 'seed')})
    assert expected_binding is not None, 'Runtime/source binding must be independently constructed'
    assert expected_binding['dense_objective'] == objective
    assert expected_binding['probe_fixture_sha256'] == fixture_entry['sha256']
    assert expected_binding['probe_case'] == setting['name']
    assert report['checkpoint_binding'] == expected_binding
    assert _digest(report['initial_state_sha256'])
    assert '4090' in report['gpu']['name'] and report['gpu']['uuid']
    assert _integer(report['gpu']['total_bytes'], 1)
    assert _finite(report['setup_seconds'])
    assert report['versions'] and isinstance(report['versions'], dict)
    validate_training_steps(report, setting, examples, objective)
    count = len(report['steps'])
    failure = report['status'] == 'cuda_oom'
    validate_resource_report(report, expected_checks=1 + count + int(failure))
    first = WARMUP + 1 if report['resumed'] else 1
    stages = ['after_runtime'] + [f'update{i}' for i in range(first, first + count)]
    if failure:
        stages.append('failure')
    assert [row['stage'] for row in report['memory']] == stages
    for snapshot in report['memory']:
        validate_memory_snapshot(snapshot)
    assert all(row['peak_gpu_reserved_bytes'] <= report['gpu']['total_bytes'] for row in report['steps'])
    if failure:
        assert not report['resumed'] and report['failure_stage'] == 'training_attempt'
        assert 'OutOfMemoryError' in report['error']
        assert 'final_state_sha256' not in report
        return

    from src.baselines.liger_prefix_training import check_progress
    assert report['train_seen_rows'] == 11542755 and report['cold_rows'] == 23915712
    progress = report['final_progress']
    budget = {key: config['training'][key] for key in ('max_optimizer_steps', 'max_attempts', 'batch_size')}
    budget['prefixes_per_epoch'] = manifest['counts']['prefix_examples_per_pass']
    check_progress(progress, budget)
    assert progress['updates'] == progress['attempts'] == progress['next_batch'] == ATTEMPTS
    assert progress['skipped_updates'] == 0 and progress['epoch'] == 0
    assert progress['prefix_count'] == progress['samples_in_epoch'] == ATTEMPTS * 1024
    assert progress['final_count'] == sum(e.is_final for e in examples)
    assert progress['dense_objective'] == objective
    assert progress['best'] is None and progress['validations'] == []
    assert _digest(report['final_state_sha256'])
    checkpoint_key = 'loaded_checkpoint' if report['resumed'] else 'resume_checkpoint'
    entry = report[checkpoint_key]
    assert isinstance(entry['path'], str) and _digest(entry['sha256'])
    timing_key = 'checkpoint_load_seconds' if report['resumed'] else 'diagnostic_checkpoint_write_seconds'
    assert _finite(report[timing_key])


def require_exact_resume(continuous, resumed):
    assert continuous['status'] == resumed['status'] == 'passed'
    assert continuous['resumed'] is False and resumed['resumed'] is True
    for key in ('gpu', 'versions', 'initial_state_sha256', 'checkpoint_binding',
                'final_state_sha256', 'final_progress', 'fixture_receipt'):
        assert continuous[key] == resumed[key], f'Resume mismatch: {key}'
    assert [row['training'] for row in continuous['steps'][WARMUP:]] == [
        row['training'] for row in resumed['steps']]
    assert continuous['resume_checkpoint'] == resumed['loaded_checkpoint']


def summarize_case(report, prefix_count=45996113):
    assert report['status'] == 'passed' and report['resumed'] is False
    measured = [row for row in report['steps'] if row['measured']]
    assert len(measured) == ATTEMPTS - WARMUP == 10
    seconds = [row['training_wall_seconds'] for row in measured]
    assert all(_finite(value, minimum=1e-30) for value in seconds)
    mean = statistics.mean(seconds)
    batches = math.ceil(prefix_count / report['batch_size'])
    return dict(name=report['name'], num_negatives=report['num_negatives'], batch_size=report['batch_size'],
        measured_steps=len(measured), warmup_steps=WARMUP,
        mean_step_seconds=mean, median_step_seconds=statistics.median(seconds),
        min_step_seconds=min(seconds), max_step_seconds=max(seconds),
        prefix_examples_per_second=report['batch_size'] / mean,
        max_gpu_allocated_bytes=max(row['peak_gpu_allocated_bytes'] for row in report['steps']),
        max_gpu_reserved_bytes=max(row['peak_gpu_reserved_bytes'] for row in report['steps']),
        max_process_self_hwm_bytes=max(row['high_water_rss_bytes'] for row in report['memory']),
        setup_seconds=report['setup_seconds'],
        diagnostic_checkpoint_write_seconds=report['diagnostic_checkpoint_write_seconds'],
        prefix_examples_per_pass=prefix_count, prefix_batches_per_pass=batches,
        indicative_single_prefix_pass_training_hours=batches * mean / 3600,
        timing_scope='ten synchronized training steps after two warmup steps, preloaded real prefix inputs; '
            'includes sampling, feature fetch, forward/backward and optimizer; excludes input streaming, '
            'runtime initialization, validation, checkpoint I/O and recovery; mean applied also to tail batch; '
            'not end-to-end or convergence time',
        memory_scope='CUDA allocated/reserved maxima over all twelve attempts; CPU process-self lifetime '
            'RSS high-water, not whole-job/cgroup RAM')


def _expected_binding(m, setting, fixture_entry):
    from scripts.liger_training_manifest import checkpoint_binding
    from src.baselines.liger_prefix_runtime import SUPERVISION_TAG
    from src.baselines.liger_sampled_catalog import sampling_binding
    stats = json.loads(Path(m['standardization']['path']).read_text())
    return dict(checkpoint_binding(m), standardization_sha256=m['standardization']['sha256'],
        standardization_stats_sha256=stats['stats_sha256'],
        standardization_seen_rows_sha256=stats['seen_rows_sha256'], supervision_tag=SUPERVISION_TAG,
        dense_objective=sampling_binding({key: setting[key] for key in ('num_negatives', 'seed')}),
        probe_fixture_sha256=fixture_entry['sha256'], probe_case=setting['name'])


def _load_pair(path, m, c, setting, examples, fixture_entry):
    reports, entries = [], []
    expected = _expected_binding(m, setting, fixture_entry)
    for suffix, resumed in (('continuous', False), ('resumed', True)):
        target = path.parent / f"{setting['name']}_{suffix}.json"
        report = json.loads(target.read_text())
        assert report['resumed'] is resumed
        validate_report(report, manifest=m, config=c, setting=setting, examples=examples,
            manifest_sha256=sha(path), fixture_entry=fixture_entry, job_id=os.environ['SLURM_JOB_ID'],
            expected_binding=expected)
        reports.append(report)
        entries.append(dict(path=str(target), sha256=sha(target)))
    return reports, entries


def case(path, name):
    from scripts.liger_sampled_probe_manifest import binding
    from scripts.probe_liger_sampled import load_fixture, deadline
    m, c = binding(path)
    fixture, examples, fixture_entry = load_fixture(path, m)
    validate_fixture(fixture, examples, asset_rows=m['asset_rows'], expected_prefixes=12288, batch_size=1024)
    setting = next(row for row in c['cases'] if row['name'] == name)
    (continuous, resumed), entries = _load_pair(path, m, c, setting, examples, fixture_entry)
    summary = None
    if continuous['status'] == 'passed':
        require_exact_resume(continuous, resumed)
        entry = continuous['resume_checkpoint']
        assert Path(entry['path']).resolve().is_relative_to(path.parent.resolve())
        assert sha(entry['path']) == entry['sha256']
        entries.append(entry)
        summary = summarize_case(continuous, m['counts']['prefix_examples_per_pass'])
        status = 'passed'
    elif continuous['status'] == 'cuda_oom':
        assert resumed['status'] == 'skipped_after_cuda_oom'
        status = 'capacity_oom'
    else:
        assert name == 'n32768' and resumed['status'] == 'skipped_after_cuda_oom'
        prior = json.loads((path.parent / 'n8192_acceptance.json').read_text())
        assert prior['status'] == 'capacity_oom'
        assert prior['manifest_sha256'] == sha(path) and prior['job_id'] == os.environ['SLURM_JOB_ID']
        status = 'skipped_after_cuda_oom'
    deadline(fixture)
    write_new(path.parent / f'{name}_acceptance.json', dict(status=status, name=name,
        job_id=os.environ['SLURM_JOB_ID'], source_git_commit=m['source_git_commit'],
        manifest_sha256=sha(path), reports=entries, fixture_receipt=fixture_entry,
        fresh_process_exact_resume=status == 'passed', summary=summary,
        **{key: False for key in FLAGS}))
    print(f'[liger-sampled] {name}: {status}; no formal training authority', flush=True)


def accept(path):
    from scripts.liger_sampled_probe_manifest import binding
    from scripts.probe_liger_sampled import load_fixture, deadline
    m, c = binding(path)
    fixture, examples, fixture_entry = load_fixture(path, m)
    validate_fixture(fixture, examples, asset_rows=m['asset_rows'], expected_prefixes=12288, batch_size=1024)
    continuous_reports, entries, receipts = [], [fixture_entry], []
    for setting in c['cases']:
        target = path.parent / f"{setting['name']}_acceptance.json"
        receipt = json.loads(target.read_text())
        assert receipt['name'] == setting['name'] and receipt['fixture_receipt'] == fixture_entry
        assert receipt['source_git_commit'] == m['source_git_commit']
        assert receipt['manifest_sha256'] == sha(path) and receipt['job_id'] == os.environ['SLURM_JOB_ID']
        assert all(receipt[key] is False for key in FLAGS)
        reports, report_entries = _load_pair(path, m, c, setting, examples, fixture_entry)
        for artifact in receipt['reports']:
            assert sha(artifact['path']) == artifact['sha256']
        assert receipt['reports'][:2] == report_entries
        continuous, resumed = reports
        if receipt['status'] == 'passed':
            require_exact_resume(continuous, resumed)
            assert receipt['fresh_process_exact_resume'] is True
            assert receipt['reports'][2:] == [continuous['resume_checkpoint']]
            assert receipt['summary'] == summarize_case(continuous, m['counts']['prefix_examples_per_pass'])
        else:
            assert receipt['fresh_process_exact_resume'] is False and receipt['summary'] is None
            assert receipt['reports'] == report_entries and resumed['status'] == 'skipped_after_cuda_oom'
            if receipt['status'] == 'capacity_oom':
                assert continuous['status'] == 'cuda_oom'
            else:
                assert receipt['status'] == 'skipped_after_cuda_oom' and setting['name'] == 'n32768'
                assert receipts[0]['status'] == 'capacity_oom'
                assert continuous['status'] == 'skipped_after_cuda_oom'
        continuous_reports.append(continuous)
        receipts.append(receipt)
        entries.append(dict(path=str(target), sha256=sha(target)))
    initialized = [r for r in continuous_reports if r['status'] != 'skipped_after_cuda_oom']
    assert initialized
    for report in initialized[1:]:
        for key in ('gpu', 'versions', 'initial_state_sha256', 'fixture_receipt'):
            assert report[key] == initialized[0][key]
    passed = all(receipt['status'] == 'passed' for receipt in receipts)
    deadline(fixture)
    write_new(path.parent / 'sampled_probe_acceptance.json', dict(
        status='passed_resource_probe' if passed else 'capacity_limited_resource_probe',
        job_id=os.environ['SLURM_JOB_ID'], source_git_commit=m['source_git_commit'],
        manifest_sha256=sha(path), reports=entries, gpu=initialized[0]['gpu'],
        summaries=[r['summary'] for r in receipts if r['status'] == 'passed'],
        capacity={r['name']: r['status'] for r in receipts}, data_preparation=fixture['data_batches'],
        cpu_ceiling=4, library_threads=1, formal_budget_approved=False,
        formal_negative_count_selected=False, **{key: False for key in FLAGS},
        next_action='author reviews measured resource costs before any learning probe or formal budget'))
    print('[liger-sampled] resource probe accepted; no automatic training or budget selection', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['case', 'accept'])
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--case', choices=['n8192', 'n32768'])
    args = parser.parse_args()
    if args.mode == 'case':
        if args.case is None:
            parser.error('--case is required in case mode')
        case(args.manifest.resolve(), args.case)
    else:
        accept(args.manifest.resolve())
