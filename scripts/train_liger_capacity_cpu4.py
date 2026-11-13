"""Fresh 384-wide, one-pass training only; validation is a separate GPU job.

The mathematical update is the already accepted logical-batch implementation.
This wrapper adds an immutable formal purpose, bounded streaming, checkpoint
publication and an independent CPU acceptance audit. No resume or second-pass
CLI is exposed. A completed training cursor awaits validation, not selection.
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

from scripts import liger_capacity_training_manifest as contract
from scripts.liger_capacity_runtime import logical_step
from scripts.probe_liger_cpu4 import sha, write_new
from scripts.train_liger_sampled_cpu4 import save_bound_checkpoint, restore_bound_checkpoint


def progress_budget(c):
    return {key: c['training'][key] for key in (
        'max_optimizer_steps', 'max_attempts', 'batch_size', 'prefixes_per_epoch')}


def build_context(m, *, devices=1):
    """Build fresh weights; inference callers explicitly load a bound model later."""
    import numpy as np
    import torch
    from scripts.liger_capacity_runtime import build_context as build_resource_context
    from src.baselines.liger_capacity import CAPACITY384_PARAMETERS, CAPACITY384_BACKBONE_PARAMETERS
    c = json.loads(Path(m['formal_config']['path']).read_text())
    resource = json.loads(Path(m['capacity_config']['path']).read_text())
    assert c['model'] == resource['model']
    assert c['training']['physical_batch_size'] == 1024
    ctx = build_resource_context(m, resource, microbatch=1024, devices=devices)
    assert ctx.initial_sha == m['expected_initial_state_sha256'], 'Fresh 384 initialization changed'
    assert ctx.parameter_report['total_params'] == CAPACITY384_PARAMETERS
    assert ctx.parameter_report['backbone_params'] == CAPACITY384_BACKBONE_PARAMETERS
    assert ctx.parameter_report['parameter_tensors'] == 152
    for key in ('batch_size', 'shuffle_buffer', 'prefix_shuffle_buffer', 'history_items',
                'max_optimizer_steps', 'max_attempts', 'warmup_steps'):
        assert ctx.config['training'][key] == c['training'][key], key
    assert ctx.config['optimizer'] == c['optimizer']
    assert ctx.config['dense_sampling'] == c['dense_sampling']
    ctx.formal_config = c
    ctx.binding = contract.checkpoint_binding(m)
    assert ctx.binding['standardization_sha256'] == m['standardization']['sha256']
    assert ctx.binding['standardization_stats_sha256'] == ctx.standardization['stats_sha256']
    assert ctx.binding['standardization_seen_rows_sha256'] == ctx.standardization['seen_rows_sha256']
    ctx.versions = dict(torch=torch.__version__, tensorflow=ctx.tf.__version__,
                        cuda=torch.version.cuda, numpy=np.__version__)
    return ctx


def verify_asset_bytes(m):
    """One sequential verification before formal initialization, outside step timing."""
    begin = time.monotonic()
    files = {}
    for entry in m['assets'].values():
        print('[liger-capacity-train] verifying frozen asset '+entry['path'], flush=True)
        assert sha(entry['path']) == entry['sha256'], 'Frozen asset bytes changed'
        files[entry['path']] = entry['sha256']
    return dict(files=files, seconds=time.monotonic()-begin)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _finite(value, minimum=0.):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def _digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def next_learning_rate(updates, c):
    t = c['training']
    step = min(updates+1, t['max_optimizer_steps'])
    scale = step/t['warmup_steps'] if step <= t['warmup_steps'] else .5*(1+math.cos(
        math.pi*(step-t['warmup_steps'])/(t['max_optimizer_steps']-t['warmup_steps'])))
    return c['optimizer']['learning_rate']*scale


def audit_step_log(path, c, *, seen_rows=11542755):
    """Independently recompute cursor, schedule, microbatch and sampling evidence."""
    from src.baselines.liger_sampled_catalog import sampling_binding
    t, objective = c['training'], sampling_binding(c['dense_sampling'])
    count = finals = attempts = updates = skips = 0
    chain, checkpoints = '0'*64, {}
    seconds = dict(input_seconds=0., training_seconds=0., total_step_seconds=0.)
    allocated = reserved = 0
    with Path(path).open() as handle:
        for line in handle:
            s = json.loads(line)
            attempts += 1
            expected = min(t['batch_size'], t['prefixes_per_epoch']-count)
            assert attempts <= t['batches_per_epoch'] and expected > 0
            assert all(_integer(s[key]) for key in ('epoch', 'attempt', 'batch', 'rows',
                'microbatch', 'microbatches', 'gradient_tensors', 'updates',
                'cumulative_prefixes', 'cumulative_final_targets', 'sampling_attempt'))
            assert s['epoch'] == 1 and s['attempt'] == s['batch'] == attempts
            assert s['rows'] == expected and s['microbatch'] == t['physical_batch_size']
            assert s['microbatches'] == math.ceil(expected/t['physical_batch_size'])
            assert _integer(s['final_target_count']) and s['final_target_count'] <= expected
            assert _digest(s['prefix_identity_sha256']) and _digest(s['candidate_pool_sha256'])
            assert s['dense_objective'] == objective and s['sampling_attempt'] == attempts-1
            # The real catalog greatly exceeds K; the positive union contributes
            # 1..rows forced classes in addition to the K shared negatives.
            classes = s['candidate_classes']
            assert _integer(classes, 1)
            assert min(seen_rows, objective['num_negatives']+1) <= classes <= min(
                seen_rows, objective['num_negatives']+expected)
            assert s['gradient_tensors'] == 152 and type(s['skipped']) is bool
            assert all(_finite(s[k]) for k in ('sid_loss', 'dense_loss', 'total_loss'))
            assert math.isclose(s['total_loss'], s['sid_loss']+s['dense_loss'], rel_tol=1e-12, abs_tol=1e-10)
            assert math.isclose(s['learning_rate'], next_learning_rate(updates, c), rel_tol=1e-13, abs_tol=1e-18)
            assert _finite(s['scale_before'], 1e-30) and _finite(s['scale_after'], 1e-30)
            if s['skipped']:
                assert s['scale_after'] < s['scale_before'] and s['grad_norm'] is None
                skips += 1
            else:
                assert s['scale_after'] >= s['scale_before'] and _finite(s['grad_norm'])
                updates += 1
            assert s['updates'] == updates
            count += expected
            finals += s['final_target_count']
            assert s['cumulative_prefixes'] == count and s['cumulative_final_targets'] == finals
            chain = hashlib.sha256((chain+s['prefix_identity_sha256']).encode()).hexdigest()
            assert s['data_chain'] == chain
            assert _finite(s['input_seconds']) and _finite(s['training_seconds'], 1e-30)
            assert _finite(s['seconds'], 1e-30)
            assert math.isclose(s['seconds'], s['input_seconds']+s['training_seconds'], rel_tol=1e-12, abs_tol=1e-9)
            seconds['input_seconds'] += s['input_seconds']
            seconds['training_seconds'] += s['training_seconds']
            seconds['total_step_seconds'] += s['seconds']
            assert _integer(s['peak_allocated_bytes'], 1)
            assert _integer(s['peak_reserved_bytes'], s['peak_allocated_bytes'])
            allocated = max(allocated, s['peak_allocated_bytes'])
            reserved = max(reserved, s['peak_reserved_bytes'])
            if attempts % t['checkpoint_every_batches'] == 0:
                checkpoints[attempts] = dict(attempts=attempts, updates=updates,
                    skipped_updates=skips, prefix_count=count, final_count=finals, data_chain=chain)
    assert attempts == t['batches_per_epoch'] and count == t['prefixes_per_epoch']
    assert finals == t['parents_per_epoch'] and updates > 0
    return dict(attempts=attempts, updates=updates, skipped_updates=skips,
        prefix_count=count, final_count=finals, data_chain=chain, **seconds,
        peak_allocated_bytes=allocated, peak_reserved_bytes=reserved, checkpoint_cursors=checkpoints)


def check_training_complete(p, c):
    from src.baselines.liger_prefix_training import check_progress
    check_progress(p, progress_budget(c))
    t = c['training']
    assert p['epoch'] == 0 and p['next_batch'] == t['batches_per_epoch']
    assert p['samples_in_epoch'] == p['prefix_count'] == t['prefixes_per_epoch']
    assert p['final_count'] == t['parents_per_epoch'] and p['updates'] > 0
    assert p['best'] is None and p['validations'] == []


def run(path):
    import torch
    from src.baselines.liger_prefix_training import initial_progress, check_progress
    from src.baselines.liger_sampled_catalog import sampling_binding
    m, c = contract.binding(path)
    assert m['phase'] == contract.FORMAL_PHASE and m['formal_training_ready'] is True
    assert 'resume_checkpoint' not in m
    directory = Path(m['run_directory'])
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(status='failed', phase=m['phase'], job_id=os.environ['SLURM_JOB_ID'],
        source_git_commit=m['source_git_commit'], manifest_sha256=sha(path),
        test_records_read=False, automatic_successor=False, checkpoints=[], awaiting_validation=True)
    started = time.monotonic()
    try:
        report['asset_verification'] = verify_asset_bytes(m)
        ctx = build_context(m)
        t, budget = c['training'], progress_budget(c)
        p = initial_progress(max_optimizer_steps=t['max_optimizer_steps'], max_attempts=t['max_attempts'])
        report.update(initial_progress=copy.deepcopy(p), initialization=m['initialization'],
            initial_state_sha256=ctx.initial_sha, checkpoint_binding=ctx.binding,
            parameters=ctx.parameter_report, resources=ctx.resources, gpus=ctx.gpus,
            versions=ctx.versions, numpy_memory_policy=ctx.numpy_policy,
            feature_cache=ctx.feature_cache_report, setup_seconds=ctx.setup_seconds)
        write_new(directory/'preflight.json', dict(report, status='ready', config=c,
            train_seen_rows=len(ctx.seen), cold_rows=len(ctx.cold), cpu_checks=ctx.cpu_checks,
            checked_at=datetime.now().astimezone().isoformat()))

        def checkpoint(name):
            entry = save_bound_checkpoint(ctx, p, directory/'checkpoints'/name)
            report['checkpoints'].append(entry)
            print('[liger-capacity-train] checkpoint '+json.dumps(entry), flush=True)
            return entry

        stream = ctx.batches(training=True, epoch=0, start_batch=0)
        try:
            with (directory/'steps.jsonl').open('x') as log:
                while True:
                    torch.cuda.synchronize()
                    begin = time.monotonic()
                    try:
                        index, records = next(stream)
                    except StopIteration:
                        break
                    input_end = time.monotonic()
                    assert index == p['next_batch'] < t['batches_per_epoch']
                    assert len(records) == min(t['batch_size'], t['prefixes_per_epoch']-p['samples_in_epoch']) > 0
                    torch.cuda.reset_peak_memory_stats()
                    step = logical_step(ctx, records, p, t['physical_batch_size'])
                    torch.cuda.synchronize()
                    ended = time.monotonic()
                    step.update(epoch=1, batch=p['next_batch'], sampling_attempt=p['attempts']-1,
                        dense_objective=sampling_binding(c['dense_sampling']),
                        input_seconds=input_end-begin, training_seconds=ended-input_end, seconds=ended-begin,
                        cumulative_prefixes=p['prefix_count'], cumulative_final_targets=p['final_count'],
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
                    check_progress(p, budget)
                    log.write(json.dumps(step, allow_nan=False)+'\n')
                    log.flush()
                    if p['next_batch'] == 1 or p['next_batch'] % 25 == 0:
                        ctx.cpu_check()
                        print('[liger-capacity-train] '+json.dumps(step), flush=True)
                    if p['next_batch'] % t['checkpoint_every_batches'] == 0:
                        checkpoint(f'pass01-batch{p["next_batch"]:06d}.pt')
        finally:
            stream.close()
        check_training_complete(p, c)
        final = checkpoint('pass01-before-validation.pt')
        ctx.cpu_check()
        contract.binding(path)
        report.update(status='passed', final_progress=copy.deepcopy(p), checkpoint=final,
            cpu_checks=ctx.cpu_checks, elapsed_seconds=time.monotonic()-started,
            completed_training_passes=1, completed_validation_passes=0)
    except Exception:
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['checked_at'] = datetime.now().astimezone().isoformat()
        write_new(directory/'training_result.json', report)


def assert_finite_tensors(value):
    import torch
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            assert torch.isfinite(value).all().item(), 'Nonfinite checkpoint tensor'
    elif isinstance(value, dict):
        for item in value.values(): assert_finite_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value: assert_finite_tensors(item)


def audit_checkpoint(entry, expected_path, expected_progress, c, binding):
    import torch
    from src.baselines.liger_prefix_training import check_progress
    from src.baselines.liger_state import tree_digest
    assert Path(entry['path']).resolve() == Path(expected_path).resolve()
    assert sha(entry['path']) == entry['sha256']
    state = torch.load(entry['path'], map_location='cpu', weights_only=False)
    assert state['format_version'] == 1 and state['binding'] == binding
    assert tree_digest(state) == entry['full_state_sha256']
    assert_finite_tensors(state['model'])
    assert_finite_tensors(state['optimizer'])
    p, t = state['progress'], c['training']
    check_progress(p, progress_budget(c))
    assert p['epoch'] == 0 and p['best'] is None and p['validations'] == []
    assert all(p[key] == value for key, value in expected_progress.items())
    for key in ('epoch', 'next_batch', 'updates', 'attempts'): assert entry[key] == p[key]
    assert state['scheduler'] == dict(total=t['max_optimizer_steps'], warmup=t['warmup_steps'],
        base_lr=c['optimizer']['learning_rate'], updates=p['updates'])
    assert state['model_training'] is True and len(state['rng']['cuda']) == 1
    assert _finite(state['scaler']['scale'], 1e-30)
    optimizer = state['optimizer']
    assert len(optimizer['param_groups']) == 1
    group = optimizer['param_groups'][0]
    assert group['weight_decay'] == c['optimizer']['weight_decay']
    assert math.isclose(group['lr'], next_learning_rate(p['updates'], c), rel_tol=1e-13, abs_tol=1e-18)
    assert len(optimizer['state']) == (152 if p['updates'] else 0)
    assert len(group['params']) == 152 and len(set(group['params'])) == 152
    assert set(optimizer['state']) <= set(group['params'])
    assert all(int(value['step']) == p['updates'] for value in optimizer['state'].values())
    return p


def accept_cpu():
    import torch
    from src.runtime_cpu import configure_torch
    from src.runtime_export_cpu import check_export_allocation, export_cpu_snapshot
    configure_torch(torch)
    resources = check_export_allocation()
    return resources, lambda: export_cpu_snapshot(resources['allowed_cpus'])


def accept(path):
    """CPU-only: reconstruct formal counters and audit every published checkpoint."""
    from src.baselines.liger_prefix_training import initial_progress
    from scripts.accept_liger_sampled_scaling import validate_numpy_policy
    m, c = contract.binding(path)
    assert m['phase'] == contract.FORMAL_PHASE and m['formal_training_ready'] is True
    acceptance_resources, cpu_check = accept_cpu()
    acceptance_checks = [cpu_check()]
    directory = Path(m['run_directory'])
    result_path = directory/'training_result.json'
    r = json.loads(result_path.read_text())
    assert r['status'] == 'passed' and r['job_id'] == os.environ['SLURM_JOB_ID']
    assert r['source_git_commit'] == m['source_git_commit'] and r['manifest_sha256'] == sha(path)
    assert r['awaiting_validation'] is True and not r['test_records_read'] and not r['automatic_successor']
    assert r['completed_training_passes'] == 1 and r['completed_validation_passes'] == 0
    expected_binding = contract.checkpoint_binding(m)
    assert r['checkpoint_binding'] == expected_binding
    assert r['initial_state_sha256'] == m['expected_initial_state_sha256']
    t = c['training']
    assert r['initial_progress'] == initial_progress(max_optimizer_steps=t['max_optimizer_steps'], max_attempts=t['max_attempts'])
    assert r['parameters']['total_params'] == r['parameters']['trainable_params'] == 41868160
    assert r['parameters']['backbone_params'] == 34222464 and r['parameters']['parameter_tensors'] == 152
    assert r['parameters']['tied_sid_input_output'] is True
    assert r['resources']['allocated_cpus'] == r['resources']['user_cpu_ceiling'] == 4
    assert r['resources']['library_threads'] == 1 and 1 <= len(r['resources']['allowed_cpus']) <= 4
    assert r['cpu_checks'] and all(x['allowed_cpus'] == r['resources']['allowed_cpus']
        and x['processes'] and all(_integer(p['threads_checked'], 1) for p in x['processes']) for x in r['cpu_checks'])
    assert len(r['gpus']) == 1 and '4090' in r['gpus'][0]['name'] and r['gpus'][0]['uuid']
    validate_numpy_policy(r['numpy_memory_policy'], r['versions']['numpy'])
    cache = r['feature_cache']
    assert cache['verification']['status'] == 'passed' and cache['verification']['feature_values_bitwise_equal']
    assert cache['build']['cache_sha256'] == cache['verification']['cache_sha256'] == m['expected_feature_cache_sha256']
    assert cache['build']['cache_bytes'] == cache['verification']['cache_bytes'] == 5909890560
    assets = r['asset_verification']
    assert assets['files'] == {v['path']: v['sha256'] for v in m['assets'].values()}
    assert _finite(assets['seconds'], 1e-30)
    measured = audit_step_log(directory/'steps.jsonl', c)
    p = r['final_progress']
    check_training_complete(p, c)
    for key in ('attempts', 'updates', 'skipped_updates', 'prefix_count', 'final_count', 'data_chain'):
        assert p[key] == measured[key]
    assert measured['peak_reserved_bytes'] <= r['gpus'][0]['total_bytes']
    intervals = list(range(t['checkpoint_every_batches'], t['batches_per_epoch']+1, t['checkpoint_every_batches']))
    names = [f'pass01-batch{batch:06d}.pt' for batch in intervals]+['pass01-before-validation.pt']
    assert [Path(e['path']).name for e in r['checkpoints']] == names
    for i, entry in enumerate(r['checkpoints']):
        if i < len(intervals):
            expected = dict(measured['checkpoint_cursors'][intervals[i]], next_batch=intervals[i])
        else:
            expected = p
        actual = audit_checkpoint(entry, directory/'checkpoints'/names[i], expected, c, expected_binding)
        if i == len(intervals): assert actual == p
    assert r['checkpoint'] == r['checkpoints'][-1]
    contract.binding(path)
    acceptance_checks.append(cpu_check())
    counts = {key: measured[key] for key in ('attempts', 'updates', 'skipped_updates', 'prefix_count', 'final_count', 'data_chain')}
    receipt = dict(status='passed_training', job_id=r['job_id'], source_git_commit=m['source_git_commit'],
        manifest_sha256=sha(path), report=dict(path=str(result_path), sha256=sha(result_path)),
        steps=dict(path=str(directory/'steps.jsonl'), sha256=sha(directory/'steps.jsonl')),
        checkpoint=r['checkpoint'], counts=counts, measured=measured,
        initial_state_sha256=r['initial_state_sha256'], checkpoint_binding=expected_binding,
        asset_verification=assets, completed_training_passes=1, completed_validation_passes=0,
        acceptance_resources=acceptance_resources, acceptance_cpu_checks=acceptance_checks,
        awaiting_validation=True, test_records_read=False, automatic_successor=False,
        checked_at=datetime.now().astimezone().isoformat())
    write_new(directory/'training_acceptance.json', receipt)
    print('[liger-capacity-train] complete one-pass training accepted; awaiting separate full validation', flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['run', 'accept'])
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    (run if args.mode == 'run' else accept)(args.manifest.resolve())
