import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import train_liger_sampled_cpu4 as runner
from src.baselines.liger_prefix_training import initial_progress
from src.baselines.liger_state import BudgetSchedule, capture, tree_digest


def config():
    return dict(training=dict(batch_size=2, prefixes_per_epoch=5, parents_per_epoch=3,
        batches_per_epoch=3, final_batch_size=1, max_optimizer_steps=3, max_attempts=3,
        warmup_steps=1, checkpoint_every_batches=2),
        optimizer=dict(learning_rate=.0003), dense_sampling=dict(num_negatives=2, seed=42),
        validation=dict(records=3))


def tiny_context():
    c = config()
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
    scheduler = BudgetSchedule(optimizer, total=3, warmup=1, base_lr=.0003)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    return SimpleNamespace(model=model, optimizer=optimizer, scheduler=scheduler,
        scaler=scaler, components=(model, optimizer, scheduler, scaler), config=c,
        training_budget=dict(max_optimizer_steps=3, max_attempts=3), binding={'purpose': 'unit'})


def synthetic_steps(*, skipped=False):
    from src.baselines.liger_sampled_catalog import sampling_binding
    c = config()
    updates = count = finals = 0
    chain = '0' * 64
    result = []
    for attempt, rows in enumerate((2, 2, 1), 1):
        skip = skipped and attempt == 2
        identity = hashlib.sha256(str(attempt).encode()).hexdigest()
        chain = hashlib.sha256((chain + identity).encode()).hexdigest()
        learning_rate = runner.next_learning_rate(updates, c)
        updates += not skip
        count += rows
        finals += 1
        result.append(dict(epoch=1, attempt=attempt, batch=attempt, rows=rows,
            prefix_count=rows, final_target_count=1, batch_unique_parents=rows,
            first_user=-attempt, last_user=attempt, prefix_identity_sha256=identity,
            batch_identity_sha256=identity, dense_objective=sampling_binding(c['dense_sampling']),
            sampling_attempt=attempt-1, forced_positive_classes=rows,
            sampled_negative_classes=2, dense_candidate_classes=rows+2,
            gradient_tensors=152, skipped=skip, sid_loss=2., dense_loss=3., total_loss=5.,
            learning_rate=learning_rate, scale_before=32., scale_after=16. if skip else 32.,
            grad_norm=None if skip else .5, updates=updates,
            cumulative_prefixes=count, cumulative_final_targets=finals, data_chain=chain,
            input_seconds=.1, training_seconds=.2, seconds=.3,
            peak_allocated_bytes=10, peak_reserved_bytes=20))
    return result


def write_steps(path, steps):
    path.write_text(''.join(json.dumps(s) + '\n' for s in steps))


@pytest.mark.parametrize('skipped', [False, True])
def test_log_audit_counts_tail_skips_original_schedule_and_hash_chain(tmp_path, skipped):
    steps = synthetic_steps(skipped=skipped)
    path = tmp_path / 'steps.jsonl'
    write_steps(path, steps)
    actual = runner.audit_step_log(path, config(), seen_rows=10)
    assert actual['attempts'] == 3 and actual['prefix_count'] == 5 and actual['final_count'] == 3
    assert actual['updates'] == 3 - skipped and actual['skipped_updates'] == skipped
    assert actual['data_chain'] == steps[-1]['data_chain']
    assert actual['input_seconds'] == pytest.approx(.3)
    assert actual['training_seconds'] == pytest.approx(.6)
    assert actual['total_step_seconds'] == pytest.approx(.9)
    assert actual['checkpoint_cursors'][2]['updates'] == 2 - skipped


@pytest.mark.parametrize('change', ['tail', 'missing', 'extra', 'final_count', 'learning_rate',
    'sampling', 'sampling_attempt', 'identity', 'data_chain', 'gradient', 'loss', 'skip',
    'scale', 'updates', 'prefix_count', 'negative_count', 'seconds', 'memory', 'nan'])
def test_log_audit_rejects_inconsistent_one_pass_evidence(tmp_path, change):
    steps = synthetic_steps()
    s = steps[-1]
    if change == 'tail': s['rows'] = 2
    elif change == 'missing': steps.pop()
    elif change == 'extra': steps.append(copy.deepcopy(s))
    elif change == 'final_count': s['final_target_count'] = 0
    elif change == 'learning_rate': s['learning_rate'] = .0003
    elif change == 'sampling': s['dense_objective']['num_negatives'] = 9
    elif change == 'sampling_attempt': s['sampling_attempt'] = 0
    elif change == 'identity': s['prefix_identity_sha256'] = '0' * 64
    elif change == 'data_chain': s['data_chain'] = '0' * 64
    elif change == 'gradient': s['gradient_tensors'] = 150
    elif change == 'loss': s['total_loss'] = 6.
    elif change == 'skip': s['skipped'] = True
    elif change == 'scale': s['scale_after'] = 1.
    elif change == 'updates': s['updates'] = 2
    elif change == 'prefix_count': s['cumulative_prefixes'] = 4
    elif change == 'negative_count': s['sampled_negative_classes'] = 1
    elif change == 'seconds': s['seconds'] = .2
    elif change == 'memory': s['peak_reserved_bytes'] = 5
    else: s['training_seconds'] = float('nan')
    path = tmp_path / 'steps.jsonl'
    write_steps(path, steps)
    with pytest.raises(AssertionError): runner.audit_step_log(path, config(), seen_rows=10)


def completed_progress():
    p = initial_progress(max_optimizer_steps=3, max_attempts=3)
    p.update(next_batch=3, samples_in_epoch=5, batches_consumed=3, attempts=3,
             prefix_count=5, final_count=3, updates=3)
    return p


def validation_entry():
    return dict(epoch=1, checkpoint={'path': 'before.pt'},
                metrics={'hybrid': {'Recall@5': .2, 'NDCG@5': .1}})


def test_finish_pass_advances_once_and_selects_only_audited_full_validation():
    p = completed_progress()
    runner.finish_pass(p, config(), validation_entry())
    assert p['epoch'] == 1 and p['next_batch'] == p['samples_in_epoch'] == 0
    assert p['best'] == dict(epoch=1, checkpoint={'path': 'before.pt'}, **validation_entry()['metrics']['hybrid'])
    assert p['validations'] == [validation_entry()]
    with pytest.raises(AssertionError): runner.finish_pass(p, config(), validation_entry())


@pytest.mark.parametrize('key,value', [('final_count', 2), ('prefix_count', 4),
    ('next_batch', 2), ('best', {'epoch': 1}), ('validations', [validation_entry()])])
def test_finish_pass_rejects_partial_or_repeated_completion(key, value):
    p = completed_progress()
    p[key] = value
    with pytest.raises((AssertionError, ValueError)): runner.finish_pass(p, config(), validation_entry())


def test_release_buffers_does_not_change_any_serialized_learning_state(monkeypatch):
    ctx = tiny_context()
    p = initial_progress(**ctx.training_budget)
    ctx.model(torch.ones(2, 1)).sum().backward()
    ctx.model.predicted_embedding = ctx.model(torch.ones(2, 1))
    before = tree_digest(capture(*ctx.components, p, ctx.binding))
    calls = []
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: calls.append(True))
    runner.release_training_buffers(ctx)
    assert ctx.model.predicted_embedding is None and all(v.grad is None for v in ctx.model.parameters())
    assert before == tree_digest(capture(*ctx.components, p, ctx.binding)) and calls == [True]


def test_checkpoint_restore_is_exact_and_create_only(tmp_path):
    ctx = tiny_context()
    p = initial_progress(**ctx.training_budget)
    target = tmp_path / 'checkpoint.pt'
    entry = runner.save_bound_checkpoint(ctx, p, target)
    with pytest.raises(FileExistsError): runner.save_bound_checkpoint(ctx, p, target)
    with torch.no_grad(): ctx.model.weight.add_(1)
    restored, proof = runner.restore_bound_checkpoint(ctx, entry)
    assert restored == p and proof['full_state_exact']
    assert proof['full_state_sha256'] == entry['full_state_sha256']
    assert tree_digest(capture(*ctx.components, p, ctx.binding)) == entry['full_state_sha256']


@pytest.mark.parametrize('failure', ['bytes', 'state', 'purpose'])
def test_checkpoint_restore_rejects_wrong_bytes_state_and_purpose(tmp_path, failure):
    ctx = tiny_context()
    p = initial_progress(**ctx.training_budget)
    entry = runner.save_bound_checkpoint(ctx, p, tmp_path / 'checkpoint.pt')
    if failure == 'bytes': entry['sha256'] = '0' * 64
    elif failure == 'state': entry['full_state_sha256'] = '0' * 64
    else: ctx.binding = {'purpose': 'other'}
    with pytest.raises((AssertionError, ValueError)): runner.restore_bound_checkpoint(ctx, entry)


@pytest.mark.parametrize('change', [None, 'weights', 'optimizer', 'progress', 'rng', 'mode'])
def test_validation_wrapper_checks_full_state_and_uses_requested_records(tmp_path, monkeypatch, change):
    from src.baselines import liger_validation
    ctx = tiny_context()
    ctx.method = ctx.official = ctx.sids = ctx.content = ctx.index = None
    ctx.seen, ctx.cold = [0], [1, 2]
    checks, requests, called = [], [], []
    ctx.cpu_check = lambda: checks.append(True)
    ctx.batches = lambda **kwargs: requests.append(kwargs) or iter([])
    p = initial_progress(**ctx.training_budget)
    report, receipt = {'metrics': {}}, {'report': {}}
    def evaluate(*args):
        assert args[10] == 3 and args[11] == {'identity': 'checkpoint'}
        called.append('evaluate')
        if change == 'weights':
            with torch.no_grad(): ctx.model.weight.add_(1)
        elif change == 'optimizer': ctx.optimizer.param_groups[0]['lr'] = .1
        elif change == 'progress': p['best'] = {'x': 1}
        elif change == 'rng': torch.rand(1)
        elif change == 'mode': ctx.model.eval()
        return report
    def audit(*args):
        assert args[3] == 3 and args[4] == 2
        called.append('audit')
        return receipt
    monkeypatch.setattr(liger_validation, 'evaluate', evaluate)
    monkeypatch.setattr(liger_validation, 'audit', audit)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    if change:
        with pytest.raises(AssertionError, match='complete learning state'):
            runner.perform_validation(ctx, p, tmp_path, 3, {'identity': 'checkpoint'}, validation_limit=3)
    else:
        a, b, proof = runner.perform_validation(ctx, p, tmp_path, 3,
            {'identity': 'checkpoint'}, validation_limit=3)
        assert a is report and b is receipt and proof['full_learning_state_unchanged']
        assert proof['before_state_sha256'] == proof['after_state_sha256']
        assert checks == [True, True]
    assert called == ['evaluate', 'audit']
    assert requests == [dict(training=False, validation_limit=3)]


def test_formal_orchestrator_is_fresh_streamed_tail_then_validation_and_stops(tmp_path, monkeypatch):
    from src.baselines import liger_prefix_training
    from scripts.probe_liger_cpu4 import write_new
    ctx = tiny_context()
    ctx.method = ctx.official = ctx.sids = ctx.content = ctx.index = ctx.catalog = None
    ctx.seen, ctx.cold = [0], [1, 2]
    ctx.initial_sha, ctx.resources, ctx.gpu, ctx.versions = 'initial', {}, {}, {}
    ctx.numpy_memory_policy, ctx.feature_cache_report, ctx.setup_seconds = {}, {}, 1.
    ctx.cpu_checks = []
    ctx.cpu_check = lambda: ctx.cpu_checks.append({'checked': True})
    requests, consumed = [], []
    def batches(**kwargs):
        requests.append(kwargs)
        return iter([(0, ['a', 'b']), (1, ['c', 'd']), (2, ['e'])])
    # The production stream is a generator and always provides close().
    ctx.batches = lambda **kwargs: (value for value in batches(**kwargs))
    m = dict(formal_training_ready=True, phase='sampled_first_pass_formal', automatic_successor=False,
        run_directory=str(tmp_path / 'run'), source_git_commit='unit', initialization='fresh seed42')
    path = tmp_path / 'manifest.json'
    path.write_text('{}')
    monkeypatch.setenv('SLURM_JOB_ID', 'unit')
    monkeypatch.setattr(runner.contract, 'binding', lambda _: m)
    monkeypatch.setattr(runner, 'build_context', lambda _: ctx)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 10)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda: 20)
    def train_step(*args, dense_sampling):
        rows, p = len(args[6]), args[11]
        assert dense_sampling == ctx.config['dense_sampling']
        consumed.append(rows)
        ctx.scheduler.step()
        p['updates'] += 1
        p['attempts'] += 1
        p['batches_consumed'] += 1
        p['next_batch'] += 1
        p['prefix_count'] += rows
        p['samples_in_epoch'] += rows
        p['final_count'] += 1
        return {'rows': rows, 'batch': p['next_batch']}
    monkeypatch.setattr(liger_prefix_training, 'train_step', train_step)
    def validate(current, p, directory, expected, checkpoint):
        assert current is ctx and expected == 3 and p['next_batch'] == 3
        directory.mkdir()
        metrics = {'hybrid': {'Recall@5': .2, 'NDCG@5': .1}}
        report = dict(metrics=metrics)
        write_new(directory / 'report.json', report)
        audited = {'report': {'path': str(directory / 'report.json'), 'sha256': runner.sha(directory / 'report.json')}}
        write_new(directory / 'acceptance.json', audited)
        return report, audited, {'full_learning_state_unchanged': True}
    monkeypatch.setattr(runner, 'perform_validation', validate)
    runner.run(path)
    report = json.loads((tmp_path / 'run/result.json').read_text())
    assert report['status'] == 'passed' and report['stopped_for_author_acceptance']
    assert not report['automatic_successor'] and not report['test_records_read']
    assert consumed == [2, 2, 1] and requests == [dict(training=True, epoch=0, start_batch=0)]
    assert report['final_progress']['epoch'] == 1 and report['final_progress']['final_count'] == 3
    assert [Path(c['path']).name for c in report['checkpoints']] == [
        'pass01-batch000002.pt', 'pass01-before-validation.pt', 'pass01-accepted.pt']
    logs = [json.loads(line) for line in (tmp_path / 'run/steps.jsonl').read_text().splitlines()]
    assert all(row['seconds'] == pytest.approx(row['input_seconds'] + row['training_seconds']) for row in logs)
    with pytest.raises(FileExistsError): runner.run(path)


@pytest.mark.parametrize('mutation', ['gate', 'resume', 'successor'])
def test_formal_runner_refuses_unapproved_scope(tmp_path, monkeypatch, mutation):
    m = dict(formal_training_ready=True, phase='sampled_first_pass_formal', automatic_successor=False)
    if mutation == 'gate': m.update(formal_training_ready=False, phase='sampled_training_integration_gate')
    elif mutation == 'resume': m['resume_checkpoint'] = {'path': 'old.pt'}
    else: m['automatic_successor'] = True
    monkeypatch.setattr(runner.contract, 'binding', lambda _: m)
    with pytest.raises(AssertionError): runner.run(tmp_path / 'unused')


def completed_artifacts(tmp_path, monkeypatch):
    """Exercise the real final acceptance over small, structurally real artifacts."""
    from src.baselines import liger_validation
    from src.baselines.liger_sampled_catalog import sampling_binding
    from scripts.probe_liger_cpu4 import write_new
    from scripts.probe_liger_sampled_scaling import numpy_memory_policy
    c = config()
    directory = tmp_path / 'run'
    directory.mkdir()
    path = tmp_path / 'manifest.json'
    path.write_text('{}')
    # Inherited census deliberately has no cold_rows field, as in the real manifest.
    m = dict(formal_training_ready=True, phase='sampled_first_pass_formal', automatic_successor=False,
        run_directory=str(directory), source_git_commit='unit', expected_initial_state_sha256='a' * 64,
        expected_feature_cache_sha256='b' * 64, asset_rows=13, counts={'prefix_examples_per_pass': 5})
    ctx = SimpleNamespace(config=c, binding={'purpose': 'unit'},
        training_budget=dict(max_optimizer_steps=3, max_attempts=3), seen=np.arange(10), index=None)
    monkeypatch.setattr(runner.contract, 'binding', lambda _: m)
    monkeypatch.setattr(runner, 'build_context', lambda *a, **kw: ctx)
    monkeypatch.setenv('SLURM_JOB_ID', 'unit')
    write_steps(directory / 'steps.jsonl', synthetic_steps())
    steps = synthetic_steps()
    def progress_after(n):
        s = steps[n - 1]
        p = initial_progress(**ctx.training_budget)
        p.update(next_batch=n, attempts=n, batches_consumed=n, updates=n,
            samples_in_epoch=s['cumulative_prefixes'], prefix_count=s['cumulative_prefixes'],
            final_count=s['cumulative_final_targets'], data_chain=s['data_chain'],
            dense_objective=sampling_binding(c['dense_sampling']))
        return p
    def state(p):
        return dict(format_version=1, binding=ctx.binding, progress=copy.deepcopy(p),
            model={'weight': torch.tensor([2.])}, model_training=True,
            optimizer={'state': {i: {'step': torch.tensor(p['updates'])} for i in range(152)}},
            scheduler=dict(total=3, warmup=1, base_lr=.0003, updates=p['updates']),
            scaler={'scale': 32.}, rng={'cuda': [torch.zeros(2, dtype=torch.uint8)]})
    def checkpoint(name, value):
        from src.baselines.liger_state import save_checkpoint
        target = directory / 'checkpoints' / name
        save_checkpoint(target, value)
        p = value['progress']
        return dict(path=str(target), sha256=runner.sha(target), full_state_sha256=tree_digest(value),
            **{key: p[key] for key in ('epoch', 'next_batch', 'updates', 'attempts')})
    periodic = checkpoint('pass01-batch000002.pt', state(progress_after(2)))
    p = progress_after(3)
    before = state(p)
    before_entry = checkpoint('pass01-before-validation.pt', before)
    val_dir = directory / 'validation-pass01'
    val_dir.mkdir()
    metrics = {'hybrid': {'Recall@5': .2, 'NDCG@5': .1}}
    write_new(val_dir / 'report.json', dict(metrics=metrics, model_sha256=tree_digest(before['model'])))
    val_receipt = dict(report={'path': str(val_dir / 'report.json'), 'sha256': runner.sha(val_dir / 'report.json')},
                       metrics=metrics)
    write_new(val_dir / 'acceptance.json', val_receipt)
    proof = dict(full_learning_state_unchanged=True, before_state_sha256=tree_digest(before),
                 after_state_sha256=tree_digest(before))
    v = dict(epoch=1, checkpoint=before_entry, report=val_receipt['report'], metrics=metrics,
        acceptance={'path': str(val_dir / 'acceptance.json'), 'sha256': runner.sha(val_dir / 'acceptance.json')},
        state_preservation=proof)
    runner.finish_pass(p, c, v)
    final = state(p)
    final_entry = checkpoint('pass01-accepted.pt', final)
    r = dict(status='passed', job_id='unit', source_git_commit='unit', manifest_sha256=runner.sha(path),
        stopped_for_author_acceptance=True, test_records_read=False, automatic_successor=False,
        checkpoint_binding=ctx.binding, initial_state_sha256='a' * 64,
        initial_progress=initial_progress(**ctx.training_budget),
        resources={'allocated_cpus': 4, 'user_cpu_ceiling': 4, 'library_threads': 1, 'allowed_cpus': [0]},
        cpu_checks=[{'allowed_cpus': [0], 'processes': [{'threads_checked': 1}]}],
        gpu={'name': 'RTX 4090', 'uuid': 'unit'}, versions={'numpy': np.__version__},
        numpy_memory_policy=numpy_memory_policy(),
        feature_cache={'build': {'cache_sha256': 'b' * 64, 'cache_bytes': 5909890560},
            'verification': {'status': 'passed', 'feature_values_bitwise_equal': True,
                'cache_sha256': 'b' * 64, 'cache_bytes': 5909890560}},
        final_progress=p, best=p['best'], validations=[v],
        checkpoints=[periodic, before_entry, final_entry])
    write_new(directory / 'result.json', r)
    def audit(*args, **kwargs):
        assert args[3] == args[4] == 3 and kwargs == {'write_receipt': False}
        return val_receipt
    monkeypatch.setattr(liger_validation, 'audit', audit)
    return path, directory, r


def test_complete_formal_acceptance_uses_real_census_schema_and_checks_saved_state(tmp_path, monkeypatch):
    path, directory, r = completed_artifacts(tmp_path, monkeypatch)
    receipt = runner.accept(path)
    assert receipt['status'] == 'passed' and receipt['completed_passes'] == 1
    assert receipt['full_training_prefixes'] == 5 and receipt['full_training_parents'] == 3
    assert receipt['full_validation_records'] == 3 and receipt['full_validation_state_unchanged']
    assert receipt['requires_author_acceptance_before_next_pass'] and not receipt['automatic_successor']
    assert (directory / 'acceptance.json').is_file()


@pytest.mark.parametrize('change', ['proof_before', 'proof_after', 'checkpoint_bytes',
    'checkpoint_state', 'sampling_progress', 'optimizer_after_validation', 'best', 'final_count', 'scope'])
def test_final_acceptance_rejects_tampered_checkpoint_or_validation_evidence(tmp_path, monkeypatch, change):
    path, directory, r = completed_artifacts(tmp_path, monkeypatch)
    v = r['validations'][0]
    if change == 'proof_before': v['state_preservation']['before_state_sha256'] = '0' * 64
    elif change == 'proof_after': v['state_preservation']['after_state_sha256'] = '0' * 64
    elif change == 'checkpoint_bytes': r['checkpoints'][0]['sha256'] = '0' * 64
    elif change == 'checkpoint_state': r['checkpoints'][0]['full_state_sha256'] = '0' * 64
    elif change == 'sampling_progress': r['final_progress']['dense_objective']['num_negatives'] = 1
    elif change == 'optimizer_after_validation':
        entry = r['checkpoints'][-1]
        state = torch.load(entry['path'], map_location='cpu', weights_only=False)
        state['optimizer']['state'][0]['unexpected'] = torch.tensor(1)
        torch.save(state, entry['path'])
        entry['sha256'], entry['full_state_sha256'] = runner.sha(entry['path']), tree_digest(state)
    elif change == 'best': r['best'] = {'epoch': 9}
    elif change == 'final_count': r['final_progress']['final_count'] = 2
    else: r['test_records_read'] = True
    (directory / 'result.json').write_text(json.dumps(r))
    with pytest.raises((AssertionError, ValueError)):
        runner.accept(path)
