"""One bounded pass of a fresh, globally scheduled 384-wide experiment.

Epochs never submit one another. A later epoch restores the preceding accepted
post-validation checkpoint, including its single-GPU RNG and stopping history.
Interrupted epochs use explicit, immutable segments; an old segment's uncommitted
tail remains on disk but is excluded from the resumed learning trajectory.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import traceback

from scripts.probe_liger_cpu4 import sha, write_new
from scripts.liger_capacity_runtime import logical_step
from scripts.train_liger_sampled_cpu4 import save_bound_checkpoint, restore_bound_checkpoint
from scripts.train_liger_capacity_cpu4 import (
    verify_asset_bytes, next_learning_rate, assert_finite_tensors, accept_cpu,
    _integer, _finite, _digest,
)


def _contract():
    from scripts import liger_capacity_multiepoch_manifest
    return liger_capacity_multiepoch_manifest


def entry(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def read_bound(reference):
    assert sha(reference['path']) == reference['sha256'], 'Bound artifact changed'
    return json.loads(Path(reference['path']).read_text())


def progress_budget(c):
    return {key: c['training'][key] for key in (
        'max_optimizer_steps', 'max_attempts', 'batch_size', 'prefixes_per_epoch')}


def new_progress(c):
    from src.baselines.liger_prefix_training import initial_progress
    from src.baselines.liger_early_stopping import initial_policy
    p = initial_progress(max_optimizer_steps=c['training']['max_optimizer_steps'],
                         max_attempts=c['training']['max_attempts'])
    p['early_stopping'] = initial_policy()
    return p


def check_progress_state(p, c):
    from src.baselines.liger_prefix_training import check_progress
    from src.baselines.liger_early_stopping import validate_policy
    check_progress(p, progress_budget(c))
    policy = validate_policy(p['early_stopping'], c)
    assert policy['completed_epochs'] == p['epoch']
    assert p['best'] == policy['best'] and p['validations'] == policy['validations']
    assert p['final_count'] >= p['epoch'] * c['training']['parents_per_epoch']
    return p


def build_context(m, *, devices=1):
    """Keep the accepted update/cache math; replace only global schedule/binding."""
    import numpy as np
    import torch
    from scripts.liger_capacity_runtime import build_context as resource_context
    from src.baselines.liger_state import BudgetSchedule
    from src.baselines.liger_prefix_runtime import training_budget
    c = json.loads(Path(m['formal_config']['path']).read_text())
    resource = json.loads(Path(m['capacity_config']['path']).read_text())
    assert c['model'] == resource['model']
    ctx = resource_context(m, resource, microbatch=1024, devices=devices)
    assert ctx.initial_sha == m['expected_initial_state_sha256']
    assert ctx.parameter_report['total_params'] == ctx.parameter_report['trainable_params'] == 41868160
    assert ctx.parameter_report['backbone_params'] == 34222464
    assert ctx.parameter_report['parameter_tensors'] == 152
    # Resource construction has its historical one-pass schedule. No update has
    # happened yet; replacing it here cannot alter initialization or optimizer.
    assert not ctx.optimizer.state and ctx.scheduler.updates == 0
    assert ctx.config == c
    t = c['training']
    ctx.scheduler = BudgetSchedule(ctx.optimizer, total=t['max_optimizer_steps'],
        warmup=t['warmup_steps'], base_lr=c['optimizer']['learning_rate'])
    ctx.training_budget = training_budget(c)
    ctx.formal_config = c
    ctx.binding = _contract().checkpoint_binding(m)
    ctx.versions = dict(torch=torch.__version__, tensorflow=ctx.tf.__version__,
                        cuda=torch.version.cuda, numpy=np.__version__)
    return ctx


def epoch_directory(m, c, epoch):
    assert type(epoch) is int and 1 <= epoch <= c['training']['max_epochs']
    return Path(m['run_directory']) / f'epoch{epoch:02d}'


def check_training_complete(p, c, epoch):
    check_progress_state(p, c)
    t = c['training']
    assert p['epoch'] == epoch - 1 and not p['early_stopping']['stopped']
    assert p['next_batch'] == t['batches_per_epoch']
    assert p['samples_in_epoch'] == t['prefixes_per_epoch']
    assert p['prefix_count'] == epoch * t['prefixes_per_epoch']
    assert p['final_count'] == epoch * t['parents_per_epoch']
    assert p['updates'] > 0


def prior_epoch_state(path, m, c, epoch):
    """Only an accepted complete checkpoint from THIS run can start epoch > 1."""
    if epoch == 1:
        return new_progress(c), None, None
    previous = epoch_directory(m, c, epoch - 1)
    reference = entry(previous / 'acceptance.json')
    accepted = read_bound(reference)
    assert accepted['status'] == 'passed'
    assert accepted['manifest_sha256'] == sha(path)
    assert accepted['source_git_commit'] == m['source_git_commit']
    assert accepted['test_records_read'] is accepted['automatic_successor'] is False
    assert accepted['stopped'] is False
    assert accepted['epoch'] == accepted['completed_passes'] == epoch - 1
    assert accepted['checkpoint_binding'] == _contract().checkpoint_binding(m)
    cp, p = accepted['checkpoint'], accepted['next_progress']
    assert Path(cp['path']).resolve() == (previous / 'after-validation.pt').resolve()
    assert sha(cp['path']) == cp['sha256']
    assert p['early_stopping'] == accepted['policy']
    check_progress_state(p, c)
    assert p['epoch'] == epoch - 1 and p['next_batch'] == p['samples_in_epoch'] == 0
    assert p['final_count'] == (epoch - 1) * c['training']['parents_per_epoch']
    assert not p['early_stopping']['stopped']
    # Bind the complete prior report and all accepted validation references, not
    # merely an unverified stop=false field in a loose JSON file.
    prior_report = read_bound(accepted['report'])
    assert Path(accepted['report']['path']).resolve() == (previous / 'result.json').resolve()
    assert prior_report['status'] == 'passed' and prior_report['epoch'] == epoch - 1
    assert prior_report['manifest_sha256'] == sha(path) and prior_report['source_git_commit'] == m['source_git_commit']
    assert prior_report['next_progress'] == p and prior_report['next_checkpoint'] == cp
    for validation in p['validations']:
        report = read_bound(validation['report'])
        receipt = read_bound(validation['acceptance'])
        assert report['status'] == receipt['status'] == 'passed'
        assert report['records'] == receipt['records'] == c['validation']['records']
        assert report['checkpoint'] == receipt['checkpoint'] == validation['checkpoint']
        assert report['metrics'] == receipt['metrics'] == validation['metrics']
    return copy.deepcopy(p), copy.deepcopy(cp), reference


def audit_step_log(path, c, initial, *, stop_batch=None, complete=False, seen_rows=11542755):
    """Reconstruct one immutable segment, optionally up to a resumed cursor."""
    from src.baselines.liger_sampled_catalog import sampling_binding
    p = copy.deepcopy(initial)
    check_progress_state(p, c)
    t, objective = c['training'], sampling_binding(c['dense_sampling'])
    limit = t['batches_per_epoch'] if stop_batch is None else stop_batch
    assert type(limit) is int and p['next_batch'] <= limit <= t['batches_per_epoch']
    seconds = dict(input_seconds=0., training_seconds=0., total_step_seconds=0.)
    allocated = reserved = consumed = 0
    cursors = {}
    with Path(path).open() as handle:
        while p['next_batch'] < limit:
            line = handle.readline()
            assert line and line.endswith('\n'), 'Missing or incomplete committed step'
            s = json.loads(line)
            expected = min(t['batch_size'], t['prefixes_per_epoch'] - p['samples_in_epoch'])
            assert expected > 0
            assert all(_integer(s[key]) for key in ('epoch', 'attempt', 'batch', 'rows', 'microbatch',
                'microbatches', 'gradient_tensors', 'updates', 'cumulative_prefixes',
                'cumulative_final_targets', 'sampling_attempt'))
            assert s['epoch'] == p['epoch'] + 1 and s['batch'] == p['next_batch'] + 1
            assert s['attempt'] == p['attempts'] + 1 and s['sampling_attempt'] == p['attempts']
            assert s['rows'] == expected and s['microbatch'] == t['physical_batch_size']
            assert s['microbatches'] == math.ceil(expected / t['physical_batch_size'])
            assert _integer(s['final_target_count']) and s['final_target_count'] <= expected
            assert _digest(s['prefix_identity_sha256']) and _digest(s['candidate_pool_sha256'])
            assert s['dense_objective'] == objective
            assert _integer(s['candidate_classes'], 1)
            assert min(seen_rows, objective['num_negatives'] + 1) <= s['candidate_classes'] <= min(
                seen_rows, objective['num_negatives'] + expected)
            assert s['gradient_tensors'] == 152 and type(s['skipped']) is bool
            assert all(_finite(s[k]) for k in ('sid_loss', 'dense_loss', 'total_loss'))
            assert math.isclose(s['total_loss'], s['sid_loss'] + s['dense_loss'], rel_tol=1e-12, abs_tol=1e-10)
            assert math.isclose(s['learning_rate'], next_learning_rate(p['updates'], c), rel_tol=1e-13, abs_tol=1e-18)
            assert _finite(s['scale_before'], 1e-30) and _finite(s['scale_after'], 1e-30)
            if s['skipped']:
                assert s['scale_after'] < s['scale_before'] and s['grad_norm'] is None
                p['skipped_updates'] += 1
            else:
                assert s['scale_after'] >= s['scale_before'] and _finite(s['grad_norm'])
                p['updates'] += 1
            assert s['updates'] == p['updates']
            p['attempts'] += 1
            p['batches_consumed'] += 1
            p['next_batch'] += 1
            p['samples_in_epoch'] += expected
            p['prefix_count'] += expected
            p['final_count'] += s['final_target_count']
            p['data_chain'] = hashlib.sha256((p['data_chain'] + s['prefix_identity_sha256']).encode()).hexdigest()
            assert s['data_chain'] == p['data_chain']
            assert s['cumulative_prefixes'] == p['prefix_count'] and s['cumulative_final_targets'] == p['final_count']
            assert _finite(s['input_seconds']) and _finite(s['training_seconds'], 1e-30) and _finite(s['seconds'], 1e-30)
            assert math.isclose(s['seconds'], s['input_seconds'] + s['training_seconds'], rel_tol=1e-12, abs_tol=1e-9)
            for key in ('input_seconds', 'training_seconds'): seconds[key] += s[key]
            seconds['total_step_seconds'] += s['seconds']
            assert _integer(s['peak_allocated_bytes'], 1)
            assert _integer(s['peak_reserved_bytes'], s['peak_allocated_bytes'])
            allocated, reserved = max(allocated, s['peak_allocated_bytes']), max(reserved, s['peak_reserved_bytes'])
            consumed += 1
            if p['next_batch'] % t['checkpoint_every_batches'] == 0:
                cursors[p['next_batch']] = copy.deepcopy(p)
        # Only an explicit checkpoint rollback permits ignoring an old log tail.
        if stop_batch is None:
            assert not handle.read(), 'Extra steps after the bounded epoch'
    check_progress_state(p, c)
    if complete: check_training_complete(p, c, p['epoch'] + 1)
    return dict(progress=p, steps=consumed, checkpoint_cursors=cursors,
                peak_allocated_bytes=allocated, peak_reserved_bytes=reserved, **seconds)


def audit_checkpoint(reference, expected_path, expected_progress, c, binding):
    import torch
    from src.baselines.liger_state import tree_digest
    assert Path(reference['path']).resolve() == Path(expected_path).resolve()
    assert sha(reference['path']) == reference['sha256']
    state = torch.load(reference['path'], map_location='cpu', weights_only=False)
    assert state['format_version'] == 1 and state['binding'] == binding
    assert tree_digest(state) == reference['full_state_sha256']
    p, t = state['progress'], c['training']
    assert p == expected_progress
    check_progress_state(p, c)
    for key in ('epoch', 'next_batch', 'updates', 'attempts'): assert reference[key] == p[key]
    assert_finite_tensors(state['model']); assert_finite_tensors(state['optimizer'])
    assert state['scheduler'] == dict(total=t['max_optimizer_steps'], warmup=t['warmup_steps'],
        base_lr=c['optimizer']['learning_rate'], updates=p['updates'])
    assert state['model_training'] is True and len(state['rng']['cuda']) == 1
    assert _finite(state['scaler']['scale'], 1e-30)
    opt = state['optimizer']
    assert len(opt['param_groups']) == 1
    group = opt['param_groups'][0]
    assert group['weight_decay'] == c['optimizer']['weight_decay']
    assert math.isclose(group['lr'], next_learning_rate(p['updates'], c), rel_tol=1e-13, abs_tol=1e-18)
    assert len(opt['state']) == (152 if p['updates'] else 0)
    assert len(group['params']) == len(set(group['params'])) == 152
    assert set(opt['state']) <= set(group['params'])
    assert all(int(value['step']) == p['updates'] for value in opt['state'].values())
    return p


def segment_from_checkpoint(reference, directory):
    path = Path(reference['path']).resolve()
    segment = path.parent.parent
    assert path.parent.name == 'checkpoints'
    assert segment.parent == (Path(directory) / 'segments').resolve()
    assert re.fullmatch(r'segment[0-9]{3}', segment.name)
    assert path.suffix == '.pt'
    return segment


def segment_chain(final_segment, directory):
    final_segment, directory = Path(final_segment), Path(directory)
    chain, visited = [], set()
    current = final_segment
    while True:
        assert current.parent == directory / 'segments' and current not in visited
        visited.add(current)
        pre = json.loads((current / 'preflight.json').read_text())
        chain.append((current, pre))
        source = pre['resume_segment']
        if source is None: break
        previous = segment_from_checkpoint(pre['loaded_checkpoint'], directory)
        assert previous.name == source and previous.name < current.name
        current = previous
    chain.reverse()
    assert chain[0][0].name == 'segment000'
    return chain


def audit_segments(path, m, c, epoch, final_segment, *, stop_batch=None, checkpoint_checks=True):
    """Audit the canonical resumed trajectory without erasing abandoned tails."""
    directory = epoch_directory(m, c, epoch)
    expected, previous, previous_receipt = prior_epoch_state(path, m, c, epoch)
    chain = segment_chain(final_segment, directory)
    segments, checkpoints = [], []
    totals = dict(input_seconds=0., training_seconds=0., total_step_seconds=0., steps=0)
    allocated = reserved = 0
    for i, (segment, pre) in enumerate(chain):
        assert pre['status'] == 'ready' and pre['epoch'] == epoch
        assert pre['manifest_sha256'] == sha(path) and pre['source_git_commit'] == m['source_git_commit']
        assert pre['checkpoint_binding'] == _contract().checkpoint_binding(m)
        assert pre['initial_state_sha256'] == m['expected_initial_state_sha256']
        assert pre['initial_progress'] == expected
        if i == 0:
            assert pre['loaded_checkpoint'] == previous and pre['previous_acceptance'] == previous_receipt
        else:
            assert pre['loaded_checkpoint']['next_batch'] == expected['next_batch']
            if checkpoint_checks:
                audit_checkpoint(pre['loaded_checkpoint'], pre['loaded_checkpoint']['path'], expected, c, pre['checkpoint_binding'])
        limit = chain[i + 1][1]['loaded_checkpoint']['next_batch'] if i + 1 < len(chain) else stop_batch
        measured = audit_step_log(segment / 'steps.jsonl', c, expected, stop_batch=limit,
                                  complete=i == len(chain)-1 and stop_batch is None)
        expected = measured['progress']
        for key in totals: totals[key] += measured[key]
        allocated, reserved = max(allocated, measured['peak_allocated_bytes']), max(reserved, measured['peak_reserved_bytes'])
        refs = []
        for cursor, snapshot in measured['checkpoint_cursors'].items():
            cp_path = segment / 'checkpoints' / f'batch{cursor:06d}.pt'
            reference = json.loads(cp_path.with_suffix('.pt.json').read_text())
            if checkpoint_checks: audit_checkpoint(reference, cp_path, snapshot, c, pre['checkpoint_binding'])
            refs.append(reference); checkpoints.append(reference)
        segments.append(dict(preflight=entry(segment/'preflight.json'), steps=entry(segment/'steps.jsonl'),
            consumed_through_batch=expected['next_batch'], segment=str(segment), checkpoints=refs,
            ignored_tail_after_checkpoint=limit is not None))
    return dict(progress=expected, segments=segments, checkpoints=checkpoints,
                peak_allocated_bytes=allocated, peak_reserved_bytes=reserved, **totals)


def _metadata(ctx, m, c, path, epoch):
    return dict(status='failed', phase=m['phase'], epoch=epoch, job_id=os.environ['SLURM_JOB_ID'],
        source_git_commit=m['source_git_commit'], manifest_sha256=sha(path),
        test_records_read=False, automatic_successor=False, awaiting_validation=True,
        initialization=m['initialization'], initial_state_sha256=ctx.initial_sha,
        checkpoint_binding=ctx.binding, parameters=ctx.parameter_report,
        resources=ctx.resources, gpus=ctx.gpus, versions=ctx.versions,
        numpy_memory_policy=ctx.numpy_policy, feature_cache=ctx.feature_cache_report,
        setup_seconds=ctx.setup_seconds, config=c)


def run(path, epoch, resume_entry=None):
    import torch
    from src.baselines.liger_sampled_catalog import sampling_binding
    contract = _contract()
    m, c = contract.binding(path)
    assert m['phase'] == contract.FORMAL_PHASE and m['formal_training_ready'] is True
    directory = epoch_directory(m, c, epoch)
    initial, checkpoint, previous_receipt = prior_epoch_state(path, m, c, epoch)
    resume_segment = None
    if resume_entry is None:
        directory.mkdir(parents=True, exist_ok=False)
        segment = directory / 'segments/segment000'
    else:
        assert directory.is_dir() and not (directory/'training_result.json').exists()
        assert not (directory/'training_acceptance.json').exists()
        checkpoint = json.loads(Path(resume_entry).read_text())
        source = segment_from_checkpoint(checkpoint, directory)
        assert Path(resume_entry).resolve() == Path(checkpoint['path']).with_suffix('.pt.json').resolve()
        assert sha(checkpoint['path']) == checkpoint['sha256']
        measured = audit_segments(path, m, c, epoch, source, stop_batch=checkpoint['next_batch'])
        initial = measured['progress']
        audit_checkpoint(checkpoint, checkpoint['path'], initial, c, contract.checkpoint_binding(m))
        resume_segment = source.name
        existing = [int(p.name[7:]) for p in (directory/'segments').iterdir()
                    if p.is_dir() and re.fullmatch(r'segment[0-9]{3}', p.name)]
        next_index = max(existing) + 1
        assert next_index < 1000
        segment = directory / 'segments' / f'segment{next_index:03d}'
    segment.mkdir(parents=True, exist_ok=False)
    (segment / 'checkpoints').mkdir()
    started, report, ctx = time.monotonic(), None, None
    try:
        assets = verify_asset_bytes(m)
        ctx = build_context(m)
        p = copy.deepcopy(initial)
        restore_proof = None
        if checkpoint is not None:
            p, restore_proof = restore_bound_checkpoint(ctx, checkpoint)
            assert p == initial
        check_progress_state(p, c)
        assert p['epoch'] == epoch - 1 and not p['early_stopping']['stopped']
        report = _metadata(ctx, m, c, path, epoch)
        report.update(asset_verification=assets, initial_progress=copy.deepcopy(p),
            loaded_checkpoint=checkpoint, previous_acceptance=previous_receipt,
            resume_segment=resume_segment, restore_proof=restore_proof, checkpoints=[])
        write_new(segment/'preflight.json', dict(report, status='ready', cpu_checks=ctx.cpu_checks,
            train_seen_rows=len(ctx.seen), cold_rows=len(ctx.cold), checked_at=datetime.now().astimezone().isoformat()))

        def save(name):
            reference = save_bound_checkpoint(ctx, p, segment/'checkpoints'/name)
            write_new(Path(reference['path']).with_suffix('.pt.json'), reference)
            report['checkpoints'].append(reference)
            print('[liger-multiepoch] checkpoint '+json.dumps(reference), flush=True)
            return reference

        t = c['training']
        stream = ctx.batches(training=True, epoch=epoch-1, start_batch=p['next_batch'])
        try:
            with (segment/'steps.jsonl').open('x') as log:
                while True:
                    torch.cuda.synchronize(); begin = time.monotonic()
                    try: index, records = next(stream)
                    except StopIteration: break
                    input_end = time.monotonic()
                    assert index == p['next_batch'] < t['batches_per_epoch']
                    assert len(records) == min(t['batch_size'], t['prefixes_per_epoch']-p['samples_in_epoch']) > 0
                    torch.cuda.reset_peak_memory_stats()
                    step = logical_step(ctx, records, p, t['physical_batch_size'])
                    torch.cuda.synchronize(); ended = time.monotonic()
                    step.update(epoch=epoch, batch=p['next_batch'], sampling_attempt=p['attempts']-1,
                        dense_objective=sampling_binding(c['dense_sampling']),
                        input_seconds=input_end-begin, training_seconds=ended-input_end, seconds=ended-begin,
                        cumulative_prefixes=p['prefix_count'], cumulative_final_targets=p['final_count'],
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
                    check_progress_state(p, c)
                    log.write(json.dumps(step, allow_nan=False)+'\n'); log.flush()
                    if p['next_batch'] == 1 or p['next_batch'] % 25 == 0:
                        ctx.cpu_check(); print('[liger-multiepoch] '+json.dumps(step), flush=True)
                    if p['next_batch'] % t['checkpoint_every_batches'] == 0:
                        save(f'batch{p["next_batch"]:06d}.pt')
        finally:
            stream.close()
        check_training_complete(p, c, epoch)
        epoch_start, _, _ = prior_epoch_state(path, m, c, epoch)
        assert p['updates'] > epoch_start['updates'], 'Epoch had no successful update'
        final = save('before-validation.pt')
        ctx.cpu_check(); contract.binding(path)
        report.update(status='passed', final_progress=copy.deepcopy(p), checkpoint=final,
            cpu_checks=ctx.cpu_checks, elapsed_seconds=time.monotonic()-started,
            completed_training_passes=epoch, completed_validation_passes=epoch-1)
    except Exception:
        if report is None:
            report = dict(status='failed', epoch=epoch, job_id=os.environ['SLURM_JOB_ID'], manifest_sha256=sha(path))
        report['status'], report['error'] = 'failed', traceback.format_exc()
        raise
    finally:
        report['checked_at'] = datetime.now().astimezone().isoformat()
        write_new(segment/'segment_result.json', report)
    result = dict(report, final_segment=str(segment), segment_result=entry(segment/'segment_result.json'))
    write_new(directory/'training_result.json', result)
    return result


def accept(path, epoch):
    """Independent CPU reconstruction plus every canonical checkpoint audit."""
    from scripts.accept_liger_sampled_scaling import validate_numpy_policy
    contract = _contract()
    m, c = contract.binding(path)
    assert m['phase'] == contract.FORMAL_PHASE and m['formal_training_ready'] is True
    resources, cpu_check = accept_cpu()
    checks = [cpu_check()]
    directory = epoch_directory(m, c, epoch)
    r = json.loads((directory/'training_result.json').read_text())
    assert r['status'] == 'passed' and r['job_id'] == os.environ['SLURM_JOB_ID'] and r['epoch'] == epoch
    assert r['manifest_sha256'] == sha(path) and r['source_git_commit'] == m['source_git_commit']
    assert r['phase'] == m['phase'] and r['awaiting_validation'] is True
    assert r['test_records_read'] is r['automatic_successor'] is False
    assert r['completed_training_passes'] == epoch and r['completed_validation_passes'] == epoch-1
    assert r['checkpoint_binding'] == contract.checkpoint_binding(m)
    assert r['initial_state_sha256'] == m['expected_initial_state_sha256']
    assert r['parameters']['total_params'] == r['parameters']['trainable_params'] == 41868160
    assert r['parameters']['backbone_params'] == 34222464 and r['parameters']['parameter_tensors'] == 152
    assert r['parameters']['tied_sid_input_output'] is True
    segment = Path(r['final_segment'])
    assert segment.parent == directory/'segments'
    assert r['segment_result'] == entry(segment/'segment_result.json')
    sr = read_bound(r['segment_result'])
    assert all(r[k] == v for k, v in sr.items())
    measured = audit_segments(path, m, c, epoch, segment)
    assert measured['progress'] == r['final_progress']
    epoch_initial, _, _ = prior_epoch_state(path, m, c, epoch)
    assert r['final_progress']['updates'] > epoch_initial['updates'], 'Epoch had no successful update'
    check_training_complete(r['final_progress'], c, epoch)
    assert r['checkpoint'] == r['checkpoints'][-1]
    audit_checkpoint(r['checkpoint'], segment/'checkpoints/before-validation.pt', r['final_progress'], c, r['checkpoint_binding'])
    for item in measured['segments']:
        pre = read_bound(item['preflight'])
        assert pre['resources']['allocated_cpus'] == pre['resources']['user_cpu_ceiling'] == 4
        assert pre['resources']['library_threads'] == 1 and 1 <= len(pre['resources']['allowed_cpus']) <= 4
        assert len(pre['gpus']) == 1 and '4090' in pre['gpus'][0]['name']
        assert pre['cpu_checks'] and all(x['allowed_cpus'] == pre['resources']['allowed_cpus'] for x in pre['cpu_checks'])
        validate_numpy_policy(pre['numpy_memory_policy'], pre['versions']['numpy'])
        cache = pre['feature_cache']
        assert cache['verification']['status'] == 'passed' and cache['verification']['feature_values_bitwise_equal']
        assert cache['build']['cache_sha256'] == cache['verification']['cache_sha256'] == m['expected_feature_cache_sha256']
        assert cache['build']['cache_bytes'] == cache['verification']['cache_bytes'] == 5909890560
        assert pre['asset_verification']['files'] == {v['path']:v['sha256'] for v in m['assets'].values()}
        assert _finite(pre['asset_verification']['seconds'], 1e-30)
    assert measured['peak_reserved_bytes'] <= r['gpus'][0]['total_bytes']
    assert r['cpu_checks'] and all(x['allowed_cpus'] == r['resources']['allowed_cpus'] for x in r['cpu_checks'])
    checks.append(cpu_check()); contract.binding(path)
    counts = {k:r['final_progress'][k] for k in ('attempts','updates','skipped_updates','prefix_count','final_count','data_chain')}
    receipt = dict(status='passed_training', epoch=epoch, job_id=r['job_id'], source_git_commit=m['source_git_commit'],
        manifest_sha256=sha(path), report=entry(directory/'training_result.json'),
        segments=measured['segments'], checkpoint=r['checkpoint'], final_progress=r['final_progress'],
        counts=counts, measured=measured, checkpoint_binding=r['checkpoint_binding'],
        initial_state_sha256=r['initial_state_sha256'], completed_training_passes=epoch,
        completed_validation_passes=epoch-1, acceptance_resources=resources, acceptance_cpu_checks=checks,
        awaiting_validation=True, test_records_read=False, automatic_successor=False,
        checked_at=datetime.now().astimezone().isoformat())
    write_new(directory/'training_acceptance.json', receipt)
    print('[liger-multiepoch] epoch training accepted; awaiting full validation', flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['run', 'accept'])
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--resume-entry', type=Path)
    args = parser.parse_args()
    if args.mode == 'accept' and args.resume_entry is not None: parser.error('accept does not restore training')
    if args.mode == 'run': run(args.manifest.resolve(), args.epoch, args.resume_entry)
    else: accept(args.manifest.resolve(), args.epoch)
