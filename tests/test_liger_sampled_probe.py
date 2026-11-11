"""CPU-only checks for the bounded sampled-loss resource probe."""
import copy
import json
import math
import pytest

from scripts import probe_liger_sampled as probe
from scripts import accept_liger_sampled as acceptance
from scripts.probe_liger_cpu4 import sha
from src.baselines.liger_data import Record
from src.baselines.liger_supervision import expand_training_records
from src.baselines.liger_sampled_catalog import sampling_binding
from src.baselines.liger_prefix_training import initial_progress


@pytest.mark.parametrize('stage,attempt,resumed,expected', [
    ('training_attempt', 1, False, True), ('training_attempt', 12, False, True),
    ('training_attempt', 0, False, False), ('training_attempt', 13, False, False),
    ('training_attempt', True, False, False), ('training_attempt', None, False, False),
    ('training_attempt', 3, True, False), ('runtime_initialization', 1, False, False),
    ('checkpoint_restore', 3, False, False), ('post_update_checks', 3, False, False),
    ('state_fingerprint', 12, False, False),
])
def test_oom_is_capacity_only_inside_continuous_training(stage, attempt, resumed, expected):
    assert probe.capacity_oom(stage, attempt, resumed=resumed) is expected


def test_deadline_remains_shared_across_cases_and_resumes(monkeypatch):
    monkeypatch.setattr(probe.time, 'time', lambda: 100.)
    probe.deadline({'deadline_unix_seconds': 101.})
    with pytest.raises(TimeoutError):
        probe.deadline({'deadline_unix_seconds': 100.})


@pytest.mark.parametrize('change', ['job', 'manifest', 'source', 'payload'])
def test_fixture_rejects_foreign_or_modified_provenance(tmp_path, monkeypatch, change):
    monkeypatch.setenv('SLURM_JOB_ID', 'current-job')
    path = tmp_path / 'manifest.json'
    path.write_text('{}')
    payload = tmp_path / 'prefix_fixture.json'
    payload.write_text(json.dumps({'examples': []}))
    manifest = {'source_git_commit': 'bound-source', 'asset_rows': 35458467,
                'phase': 'sampled_dense_resource_and_resume_probe_only'}
    receipt = dict(status='passed', job_id='current-job', manifest_sha256=sha(path),
                   source_git_commit='bound-source', phase=manifest['phase'],
                   fixture={'path': str(payload), 'sha256': sha(payload)})
    if change == 'job': receipt['job_id'] = 'other-job'
    elif change == 'manifest': receipt['manifest_sha256'] = 'changed'
    elif change == 'source': receipt['source_git_commit'] = 'changed'
    else: payload.write_text(json.dumps({'examples': [], 'changed': True}))
    (tmp_path / 'fixture_receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(AssertionError):
        probe.load_fixture(path, manifest, check_deadline=False)


def tiny_examples():
    # Repeated item identities, distinct parent provenance, mixed prefix/final
    # labels, and shared parents across batch boundaries are all intentional.
    return expand_training_records([
        Record(i, tuple(range(8)), f'{i:064x}') for i in range(1756)
    ])[:12288]


def setting():
    return dict(name='n8192', backend='lazy', batch_size=1024,
                num_negatives=8192, seed=42)


def objective():
    return sampling_binding({'num_negatives': 8192, 'seed': 42})


def reports():
    examples = tiny_examples()
    steps = []
    for index in range(12):
        batch = examples[index * 1024:(index + 1) * 1024]
        identity = acceptance.expected_identity(batch)
        forced = len({e.sequence[-1] for e in batch})
        training = dict(batch=index + 1, rows=1024, attempt=index + 1, prefix_count=1024,
            final_target_count=sum(e.is_final for e in batch),
            batch_unique_parents=len({e.parent_user_id for e in batch}),
            batch_identity_sha256=identity, prefix_identity_sha256=identity,
            first_user=batch[0].parent_user_id, last_user=batch[-1].parent_user_id,
            updates=index + 1, skipped=False, learning_rate=.0003 * (index + 1) / 963,
            scale_before=65536., scale_after=65536., sid_loss=2., dense_loss=3., total_loss=5.,
            gradient_tensors=152, grad_norm=1., dense_objective=objective(),
            forced_positive_classes=forced, sampled_negative_classes=8192,
            dense_candidate_classes=8192 + forced, sampling_attempt=index)
        steps.append(dict(training=training, measured=index >= 2,
            training_wall_seconds=1000. if index < 2 else float(index - 1),
            catalog_fetch=dict(calls=1, rows=8192 + forced, failed_calls=0,
                               unknown_rows_calls=0, cpu_fetch_wall_seconds=.1),
            peak_gpu_allocated_bytes=100 + index, peak_gpu_reserved_bytes=200 + index))
    a = dict(status='passed', **setting(), resumed=False, steps=steps,
        gpu=dict(name='RTX 4090', uuid='same', total_bytes=24 * 1024**3),
        initial_state_sha256='initial', final_state_sha256='final', versions={'torch': 'fixture'},
        final_progress={'updates': 12, 'attempts': 12},
        checkpoint_binding={'dense_objective': objective(), 'probe_fixture_sha256': 'fixture',
                            'probe_case': 'n8192'},
        fixture_receipt={'path': '/fixture', 'sha256': 'fixture'},
        resume_checkpoint={'path': '/diagnostic-step2', 'sha256': 'checkpoint'},
        setup_seconds=20., elapsed_seconds=30., diagnostic_checkpoint_write_seconds=2.,
        memory=[{'high_water_rss_bytes': 1024}])
    b = copy.deepcopy(a)
    b.update(resumed=True, steps=b['steps'][2:],
             loaded_checkpoint=b.pop('resume_checkpoint'))
    # Timing and allocator observations are not part of learning-state identity.
    for step in b['steps']:
        step['training_wall_seconds'] *= 5.
        step['peak_gpu_allocated_bytes'] += 10
    return a, b, examples


def test_fresh_process_resume_ignores_resource_timing_but_not_learning():
    a, b, _ = reports()
    acceptance.require_exact_resume(a, b)


@pytest.mark.parametrize('change', [
    'status', 'gpu', 'initial', 'final', 'progress', 'binding', 'fixture',
    'checkpoint', 'loss', 'sample_attempt', 'missing_step',
])
def test_fresh_process_resume_rejects_any_learning_or_provenance_change(change):
    a, b, _ = reports()
    if change == 'status': b['status'] = 'failed'
    elif change == 'gpu': b['gpu']['uuid'] = 'other'
    elif change == 'initial': b['initial_state_sha256'] = 'other'
    elif change == 'final': b['final_state_sha256'] = 'other'
    elif change == 'progress': b['final_progress']['updates'] = 11
    elif change == 'binding': b['checkpoint_binding']['dense_objective']['num_negatives'] = 32768
    elif change == 'fixture': b['fixture_receipt']['sha256'] = 'other'
    elif change == 'checkpoint': b['loaded_checkpoint']['sha256'] = 'other'
    elif change == 'loss': b['steps'][0]['training']['dense_loss'] += 1e-12
    elif change == 'sample_attempt': b['steps'][0]['training']['sampling_attempt'] = 99
    else: b['steps'].pop()
    with pytest.raises(AssertionError):
        acceptance.require_exact_resume(a, b)


def test_sampled_step_validator_recomputes_targets_and_prefix_identity():
    a, b, examples = reports()
    acceptance.validate_training_steps(a, setting(), examples, objective())
    acceptance.validate_training_steps(b, setting(), examples, objective())


@pytest.mark.parametrize('change', [
    'rows', 'identity', 'parents', 'final_count', 'warmup', 'sampling_attempt',
    'objective', 'forced_gt', 'negative_count', 'candidate_count', 'fetch_rows',
    'fetch_calls', 'failed_fetch', 'nonfinite_time', 'skipped', 'gradients', 'lr',
    'loss', 'reserved_memory',
])
def test_sampled_step_validator_rejects_counter_and_objective_tampering(change):
    a, _, examples = reports()
    row = a['steps'][3]
    step = row['training']
    if change == 'rows': step['rows'] -= 1
    elif change == 'identity': step['prefix_identity_sha256'] = 'changed'
    elif change == 'parents': step['batch_unique_parents'] += 1
    elif change == 'final_count': step['final_target_count'] += 1
    elif change == 'warmup': row['measured'] = False
    elif change == 'sampling_attempt': step['sampling_attempt'] = 0
    elif change == 'objective': step['dense_objective']['num_negatives'] = 32768
    elif change == 'forced_gt': step['forced_positive_classes'] += 1
    elif change == 'negative_count': step['sampled_negative_classes'] -= 1
    elif change == 'candidate_count': step['dense_candidate_classes'] -= 1
    elif change == 'fetch_rows': row['catalog_fetch']['rows'] += 1
    elif change == 'fetch_calls': row['catalog_fetch']['calls'] = 2
    elif change == 'failed_fetch': row['catalog_fetch']['failed_calls'] = 1
    elif change == 'nonfinite_time': row['training_wall_seconds'] = float('nan')
    elif change == 'skipped': step['skipped'] = True
    elif change == 'gradients': step['gradient_tensors'] = 151
    elif change == 'lr': step['learning_rate'] *= 2
    elif change == 'loss': step['total_loss'] += 1
    else: row['peak_gpu_reserved_bytes'] = row['peak_gpu_allocated_bytes'] - 1
    with pytest.raises(AssertionError):
        acceptance.validate_training_steps(a, setting(), examples, objective())


def test_summary_uses_only_ten_continuous_postwarmup_steps_and_exact_tail():
    a, _, _ = reports()
    result = acceptance.summarize_case(a)
    assert result['measured_steps'] == 10 and result['warmup_steps'] == 2
    assert result['mean_step_seconds'] == result['median_step_seconds'] == 5.5
    assert result['min_step_seconds'] == 1. and result['max_step_seconds'] == 10.
    assert result['prefix_examples_per_second'] == 1024 / 5.5
    assert result['prefix_batches_per_pass'] == math.ceil(45996113 / 1024) == 44919
    assert result['indicative_single_prefix_pass_training_hours'] == 44919 * 5.5 / 3600
    assert result['max_gpu_allocated_bytes'] == 111
    assert result['max_gpu_reserved_bytes'] == 211
    assert result['max_process_self_hwm_bytes'] == 1024
    assert result['setup_seconds'] == 20.
    assert result['diagnostic_checkpoint_write_seconds'] == 2.
    assert 'excludes input streaming' in result['timing_scope']
    assert 'not end-to-end or convergence' in result['timing_scope']


@pytest.mark.parametrize('change', ['resumed', 'missing', 'nonfinite'])
def test_summary_cannot_present_replay_or_incomplete_steps_as_throughput(change):
    a, b, _ = reports()
    if change == 'resumed': a = b
    elif change == 'missing': a['steps'].pop()
    else: a['steps'][2]['training_wall_seconds'] = float('inf')
    with pytest.raises(AssertionError):
        acceptance.summarize_case(a)


def memory():
    return dict(schema='liger.process_self_resources.v1', errors=[], rss_bytes=100,
        high_water_rss_bytes=120, virtual_memory_bytes=200, anonymous_rss_bytes=60,
        file_rss_bytes=40, shared_rss_bytes=0, swap_bytes=0, ru_maxrss_bytes=121,
        threads=1, clock_ticks_per_second=100, monotonic_seconds=1.,
        proc_cpu_user_seconds=.2, proc_cpu_system_seconds=.3, proc_cpu_total_seconds=.5,
        resource_cpu_user_seconds=.21, resource_cpu_system_seconds=.31)


def full_report():
    r, _, examples = reports()
    manifest = dict(source_git_commit='c' * 40, phase='sampled_dense_resource_and_resume_probe_only',
                    counts={'prefix_examples_per_pass': 45996113})
    config = {'training': dict(max_optimizer_steps=19260, max_attempts=12, batch_size=1024)}
    progress = initial_progress(max_optimizer_steps=19260, max_attempts=12)
    progress.update(next_batch=12, samples_in_epoch=12288, batches_consumed=12, attempts=12,
        prefix_count=12288, final_count=sum(e.is_final for e in examples), updates=12,
        dense_objective=objective())
    r.update(manifest, manifest_sha256='d' * 64, job_id='same',
        initial_state_sha256='a' * 64, final_state_sha256='b' * 64,
        final_progress=progress, train_seen_rows=11542755, cold_rows=23915712,
        resources=dict(allocated_cpus=4, user_cpu_ceiling=4, library_threads=1,
                       job_id='same', allowed_cpus=[0, 1, 2, 3]),
        cpu_checks=[dict(allowed_cpus=[0, 1, 2, 3],
                         processes=[dict(pid=100, threads_checked=3)]) for _ in range(13)],
        memory=[dict(stage=stage, **memory()) for stage in
                ['after_runtime'] + [f'update{i}' for i in range(1, 13)]],
        **{key: False for key in acceptance.FLAGS})
    r['resume_checkpoint']['sha256'] = 'e' * 64
    kwargs = dict(manifest=manifest, config=config, setting=setting(), examples=examples,
        manifest_sha256='d' * 64, fixture_entry=copy.deepcopy(r['fixture_receipt']), job_id='same',
        expected_binding=copy.deepcopy(r['checkpoint_binding']))
    return r, kwargs


def test_full_report_validates_resources_scope_and_independent_binding():
    report, kwargs = full_report()
    acceptance.validate_report(report, **kwargs)


@pytest.mark.parametrize('change', [
    'job', 'manifest', 'source', 'formal', 'heldout', 'binding', 'cpu_budget',
    'cpu_checks', 'memory', 'progress', 'cold_catalog', 'checkpoint',
])
def test_full_report_rejects_unbound_or_unsafe_artifacts(change):
    r, kwargs = full_report()
    if change == 'job': r['job_id'] = 'other'
    elif change == 'manifest': r['manifest_sha256'] = 'other'
    elif change == 'source': r['source_git_commit'] = 'other'
    elif change == 'formal': r['formal_training_ready'] = True
    elif change == 'heldout': r['validation_test_records_read'] = True
    elif change == 'binding': r['checkpoint_binding']['dense_objective']['num_negatives'] = 32768
    elif change == 'cpu_budget': r['resources']['allocated_cpus'] = 24
    elif change == 'cpu_checks': r['cpu_checks'].pop()
    elif change == 'memory': r['memory'][1]['stage'] = 'after_runtime'
    elif change == 'progress': r['final_progress']['updates'] = 11
    elif change == 'cold_catalog': r['cold_rows'] = 100
    else: r['resume_checkpoint']['sha256'] = 'not-a-digest'
    with pytest.raises((AssertionError, ValueError)):
        acceptance.validate_report(r, **kwargs)


def oom_report():
    r, kwargs = full_report()
    r.update(status='cuda_oom', failure_stage='training_attempt', failure_attempt=4,
             error='torch.cuda.OutOfMemoryError: synthetic capacity failure')
    r.pop('final_state_sha256')
    r['steps'] = r['steps'][:3]
    r['cpu_checks'] = r['cpu_checks'][:5]
    r['memory'] = r['memory'][:4] + [dict(stage='failure', **memory())]
    return r, kwargs


def test_real_training_stage_oom_is_a_valid_capacity_result():
    r, kwargs = oom_report()
    acceptance.validate_report(r, **kwargs)


@pytest.mark.parametrize('change', ['stage', 'exception', 'resumed', 'attempt'])
def test_initialization_or_recovery_failures_cannot_be_reported_as_capacity(change):
    r, kwargs = oom_report()
    if change == 'stage': r['failure_stage'] = 'runtime_initialization'
    elif change == 'exception': r['error'] = 'FileNotFoundError'
    elif change == 'resumed': r['resumed'] = True
    else: r['failure_attempt'] = 5
    with pytest.raises(AssertionError):
        acceptance.validate_report(r, **kwargs)
