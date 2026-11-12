"""One fresh sampled-prefix pass and complete cold validation, then stop.

The shared context/validation/checkpoint helpers are also exercised by the
fresh-process integration gate. Historical adapters and artifacts are untouched.
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

from scripts import liger_sampled_training_manifest as contract
from scripts.probe_liger_cpu4 import sha, write_new


def progress_budget(config):
    training = config['training']
    return {key: training[key] for key in (
        'max_optimizer_steps', 'max_attempts', 'batch_size', 'prefixes_per_epoch')}


def build_context(m, *, build_model=True, deadline=None):
    """Fresh runtime plus a verified cache of fixed standardized INPUTS only."""
    from src.baselines.liger_prefix_runtime import PrefixRuntime
    from scripts.probe_liger_sampled_scaling import numpy_memory_policy
    from src.baselines.liger_feature_cache import build_cached_seen, verify_cached_seen
    policy = numpy_memory_policy()
    started = time.monotonic()
    ctx = PrefixRuntime(m, build_model=build_model)
    inherited = {key: ctx.binding[key] for key in (
        'standardization_sha256', 'standardization_stats_sha256',
        'standardization_seen_rows_sha256', 'supervision_tag')}
    ctx.binding = dict(contract.checkpoint_binding(m), **inherited,
                       feature_cache_sha256=m['expected_feature_cache_sha256'])
    ctx.numpy_memory_policy = policy
    ctx.feature_cache_report = None
    if not build_model:
        return ctx
    assert ctx.initial_sha == m['expected_initial_state_sha256'], 'Fresh initialization changed'
    if deadline is not None:
        deadline()
    expected = dict(standardization_stats_sha256=ctx.standardization['stats_sha256'],
        seen_rows_sha256=ctx.standardization['seen_rows_sha256'],
        source_sha256=ctx.standardization['content_sha256'])
    last = {}
    def progress(*, stage, completed_rows, total_rows):
        if deadline is not None:
            deadline()
        last[stage] = last.get(stage, 0) + 1
        if last[stage] % 16 == 0 or completed_rows == total_rows:
            print('[liger-sampled-cache] ' + json.dumps(dict(stage=stage,
                completed_rows=completed_rows, total_rows=total_rows,
                elapsed_seconds=time.monotonic() - started)), flush=True)
    begin = time.monotonic()
    content, catalog, built = build_cached_seen(ctx.content, ctx.seen,
        expected_binding=expected, chunk_size=65536, progress_callback=progress)
    build_seconds = time.monotonic() - begin
    begin = time.monotonic()
    verified = verify_cached_seen(content, catalog, expected_binding=expected,
        chunk_size=65536, progress_callback=progress)
    assert built['cache_sha256'] == verified['cache_sha256'] == m['expected_feature_cache_sha256']
    ctx.feature_cache_report = dict(build=built, verification=verified,
        build_seconds=build_seconds, verify_seconds=time.monotonic() - begin,
        scope='immutable standardized train-seen CPU input features; no learned projection')
    ctx.content, ctx.catalog = content, catalog
    ctx.setup_seconds = time.monotonic() - started
    ctx.cpu_check()
    return ctx


def release_training_buffers(ctx):
    """Discard only transient gradients/query graphs before cold GPU caching.

    These are not checkpoint state. The validation wrapper independently proves
    that model/optimizer/scheduler/scaler/RNG/progress/mode remain unchanged.
    """
    import torch
    ctx.optimizer.zero_grad(set_to_none=True)
    if hasattr(ctx.model, 'predicted_embedding'):
        ctx.model.predicted_embedding = None
    torch.cuda.empty_cache()


def save_bound_checkpoint(ctx, progress, path):
    from src.baselines.liger_state import capture, save_checkpoint, tree_digest
    path = Path(path)
    state = capture(*ctx.components, progress, ctx.binding)
    digest = tree_digest(state)
    save_checkpoint(path, state)
    return dict(path=str(path.resolve()), sha256=sha(path), full_state_sha256=digest,
        epoch=progress['epoch'], next_batch=progress['next_batch'],
        updates=progress['updates'], attempts=progress['attempts'])


def restore_bound_checkpoint(ctx, entry):
    """Restore only an explicitly hash-bound checkpoint of this exact purpose."""
    import torch
    from src.baselines.liger_state import capture, restore_checkpoint, tree_digest
    assert sha(entry['path']) == entry['sha256'], 'Checkpoint bytes changed'
    state = torch.load(entry['path'], map_location='cpu', weights_only=False)
    original = tree_digest(state)
    assert original == entry['full_state_sha256'], 'Checkpoint state fingerprint changed'
    p = restore_checkpoint(entry['path'], *ctx.components, ctx.binding)
    restored = tree_digest(capture(*ctx.components, p, ctx.binding))
    assert original == restored, 'Full checkpoint state did not restore exactly'
    return p, dict(full_state_exact=True, full_state_sha256=restored,
                   checkpoint=dict(entry))


def perform_validation(ctx, progress, directory, expected_records, checkpoint, *, validation_limit=None):
    """Full cold scoring, independent metric audit, and exact state preservation."""
    from src.baselines.liger_state import capture, tree_digest
    from src.baselines.liger_validation import evaluate, audit
    # Hash immediately: capture contains live tensor/progress references.
    before = tree_digest(capture(*ctx.components, progress, ctx.binding))
    release_training_buffers(ctx)
    ctx.cpu_check()
    evaluated = evaluate(ctx.model, ctx.method, ctx.official,
        ctx.batches(training=False, validation_limit=validation_limit),
        ctx.sids, ctx.content, ctx.index, ctx.seen, ctx.cold,
        Path(directory), expected_records, checkpoint)
    audited = audit(Path(directory), ctx.seen, ctx.index, expected_records,
                    len(ctx.cold), checkpoint)
    release_training_buffers(ctx)
    after = tree_digest(capture(*ctx.components, progress, ctx.binding))
    assert before == after, 'Validation modified complete learning state'
    ctx.cpu_check()
    return evaluated, audited, dict(full_learning_state_unchanged=True,
        before_state_sha256=before, after_state_sha256=after,
        compared='model, optimizer, scheduler, scaler, RNG, progress, binding, model training mode')


def finish_pass(progress, config, validation_entry):
    from src.baselines.liger_prefix_training import check_progress
    from src.baselines.liger_state import better_validation
    t = config['training']
    check_progress(progress, progress_budget(config))
    assert progress['epoch'] == 0 and progress['next_batch'] == t['batches_per_epoch']
    assert progress['samples_in_epoch'] == progress['prefix_count'] == t['prefixes_per_epoch']
    assert progress['final_count'] == t['parents_per_epoch']
    assert validation_entry['epoch'] == 1 and progress['validations'] == [] and progress['best'] is None
    scores = validation_entry['metrics']['hybrid']
    assert better_validation(scores, 1, None)
    progress['best'] = dict(epoch=1, checkpoint=validation_entry['checkpoint'], **scores)
    progress['validations'].append(validation_entry)
    progress.update(epoch=1, next_batch=0, samples_in_epoch=0)
    check_progress(progress, progress_budget(config))


def _finite(value, minimum=0.):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= minimum


def _integer(value, minimum=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def next_learning_rate(updates, config):
    """Independent arithmetic, not a call to the training scheduler."""
    t = config['training']
    step = min(updates + 1, t['max_optimizer_steps'])
    scale = step / t['warmup_steps'] if step <= t['warmup_steps'] else .5 * (
        1 + math.cos(math.pi * (step - t['warmup_steps']) / (t['max_optimizer_steps'] - t['warmup_steps'])))
    return config['optimizer']['learning_rate'] * scale


def audit_step_log(path, config, *, seen_rows=11542755):
    """Stream the bounded one-pass log and recompute counts, LR and hash chain."""
    from src.baselines.liger_sampled_catalog import sampling_binding
    t = config['training']
    objective = sampling_binding(config['dense_sampling'])
    count = finals = attempts = updates = skips = 0
    chain = '0' * 64
    input_seconds = training_seconds = total_seconds = 0.
    allocated = reserved = 0
    checkpoints = {}
    with Path(path).open() as handle:
        for line in handle:
            s = json.loads(line)
            attempts += 1
            expected = min(t['batch_size'], t['prefixes_per_epoch'] - count)
            assert attempts <= t['batches_per_epoch'] and expected > 0
            assert s['epoch'] == 1 and s['attempt'] == s['batch'] == attempts
            assert s['rows'] == s['prefix_count'] == expected
            assert _integer(s['final_target_count']) and s['final_target_count'] <= expected
            assert _integer(s['batch_unique_parents'], 1) and s['batch_unique_parents'] <= expected
            assert isinstance(s['first_user'], int) and isinstance(s['last_user'], int)
            assert _digest(s['prefix_identity_sha256']) and s['prefix_identity_sha256'] == s['batch_identity_sha256']
            assert s['dense_objective'] == objective and s['sampling_attempt'] == attempts - 1
            forced = s['forced_positive_classes']
            assert _integer(forced, 1) and forced <= min(expected, seen_rows)
            negative = min(objective['num_negatives'], seen_rows - forced)
            assert s['sampled_negative_classes'] == negative and s['dense_candidate_classes'] == forced + negative
            assert s['gradient_tensors'] == 152 and isinstance(s['skipped'], bool)
            assert all(_finite(s[k]) for k in ('sid_loss', 'dense_loss', 'total_loss'))
            assert math.isclose(s['total_loss'], s['sid_loss'] + s['dense_loss'], rel_tol=1e-6, abs_tol=1e-5)
            assert math.isclose(s['learning_rate'], next_learning_rate(updates, config), rel_tol=1e-13, abs_tol=1e-18)
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
            chain = hashlib.sha256((chain + s['batch_identity_sha256']).encode()).hexdigest()
            assert s['data_chain'] == chain
            assert _finite(s['input_seconds']) and _finite(s['training_seconds'], 1e-30)
            assert math.isclose(s['seconds'], s['input_seconds'] + s['training_seconds'], rel_tol=1e-12, abs_tol=1e-9)
            input_seconds += s['input_seconds']
            training_seconds += s['training_seconds']
            total_seconds += s['seconds']
            assert _integer(s['peak_allocated_bytes'], 1)
            assert _integer(s['peak_reserved_bytes'], s['peak_allocated_bytes'])
            allocated = max(allocated, s['peak_allocated_bytes'])
            reserved = max(reserved, s['peak_reserved_bytes'])
            if attempts % t['checkpoint_every_batches'] == 0:
                checkpoints[attempts] = dict(updates=updates, attempts=attempts,
                    prefix_count=count, final_count=finals, data_chain=chain)
    assert attempts == t['batches_per_epoch'] and count == t['prefixes_per_epoch']
    assert finals == t['parents_per_epoch']
    return dict(attempts=attempts, updates=updates, skipped_updates=skips,
        prefix_count=count, final_count=finals, data_chain=chain,
        input_seconds=input_seconds, training_seconds=training_seconds,
        total_step_seconds=total_seconds, peak_allocated_bytes=allocated,
        peak_reserved_bytes=reserved, checkpoint_cursors=checkpoints)


def run(path):
    import torch
    from src.baselines.liger_prefix_training import initial_progress, check_progress, train_step
    m = contract.binding(path)
    assert m['formal_training_ready'] is True and m['phase'] == 'sampled_first_pass_formal'
    assert 'resume_checkpoint' not in m and m['automatic_successor'] is False
    directory = Path(m['run_directory'])
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(status='failed', phase=m['phase'], job_id=os.environ['SLURM_JOB_ID'],
        source_git_commit=m['source_git_commit'], manifest_sha256=sha(path),
        test_records_read=False, automatic_successor=False, checkpoints=[], validations=[])
    started = time.monotonic()
    try:
        ctx = build_context(m)
        t, budget = ctx.config['training'], progress_budget(ctx.config)
        p = initial_progress(**ctx.training_budget)
        check_progress(p, budget)
        report.update(initial_progress=copy.deepcopy(p), initialization=m['initialization'],
            initial_state_sha256=ctx.initial_sha, checkpoint_binding=ctx.binding,
            resources=ctx.resources, gpu=ctx.gpu, versions=ctx.versions,
            numpy_memory_policy=ctx.numpy_memory_policy, feature_cache=ctx.feature_cache_report,
            setup_seconds=ctx.setup_seconds)
        write_new(directory / 'preflight.json', dict(report, status='ready', config=ctx.config,
            train_seen_rows=len(ctx.seen), cold_rows=len(ctx.cold), cpu_checks=ctx.cpu_checks,
            checked_at=datetime.now().astimezone().isoformat()))

        def checkpoint(name):
            entry = save_bound_checkpoint(ctx, p, directory / 'checkpoints' / name)
            report['checkpoints'].append(entry)
            print('[liger-sampled-train] checkpoint ' + json.dumps(entry), flush=True)
            return entry

        stream = ctx.batches(training=True, epoch=0, start_batch=0)
        try:
            with (directory / 'steps.jsonl').open('x') as log:
                while True:
                    torch.cuda.synchronize()
                    begin = time.monotonic()
                    try:
                        index, records = next(stream)
                    except StopIteration:
                        break
                    input_end = time.monotonic()
                    assert index == p['next_batch'] < t['batches_per_epoch']
                    assert len(records) == min(t['batch_size'], t['prefixes_per_epoch'] - p['samples_in_epoch']) > 0
                    torch.cuda.reset_peak_memory_stats()
                    step = train_step(*ctx.components, ctx.method, ctx.official, records,
                        ctx.sids, ctx.content, ctx.seen, ctx.catalog, p,
                        dense_sampling=ctx.config['dense_sampling'])
                    torch.cuda.synchronize()
                    ended = time.monotonic()
                    step.update(epoch=1, input_seconds=input_end - begin,
                        training_seconds=ended - input_end, seconds=ended - begin,
                        data_chain=p['data_chain'], cumulative_prefixes=p['prefix_count'],
                        cumulative_final_targets=p['final_count'],
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
                    check_progress(p, budget)
                    log.write(json.dumps(step, allow_nan=False) + '\n')
                    log.flush()
                    if p['next_batch'] == 1 or p['next_batch'] % 25 == 0:
                        ctx.cpu_check()
                        print('[liger-sampled-train] ' + json.dumps(step), flush=True)
                    if p['next_batch'] % t['checkpoint_every_batches'] == 0:
                        checkpoint(f'pass01-batch{p["next_batch"]:06d}.pt')
        finally:
            stream.close()
        assert p['next_batch'] == t['batches_per_epoch']
        assert p['prefix_count'] == t['prefixes_per_epoch'] and p['final_count'] == t['parents_per_epoch']
        before_val = checkpoint('pass01-before-validation.pt')
        val_dir = directory / 'validation-pass01'
        evaluated, audited, proof = perform_validation(ctx, p, val_dir,
            ctx.config['validation']['records'], before_val)
        entry = dict(epoch=1, report=audited['report'],
            acceptance=dict(path=str(val_dir / 'acceptance.json'), sha256=sha(val_dir / 'acceptance.json')),
            checkpoint=before_val, metrics=evaluated['metrics'], state_preservation=proof)
        report['validations'].append(entry)
        finish_pass(p, ctx.config, entry)
        checkpoint('pass01-accepted.pt')
        ctx.cpu_check()
        contract.binding(path)
        report.update(status='passed', final_progress=p, best=p['best'], cpu_checks=ctx.cpu_checks,
            elapsed_seconds=time.monotonic() - started, stopped_for_author_acceptance=True)
    except Exception:
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['checked_at'] = datetime.now().astimezone().isoformat()
        write_new(directory / 'result.json', report)


def accept(path):
    """Independent CPU arithmetic and artifact audit; no inference is repeated."""
    import torch
    from src.baselines.liger_prefix_training import initial_progress, check_progress
    from src.baselines.liger_state import tree_digest, better_validation
    from src.baselines.liger_sampled_catalog import sampling_binding
    from src.baselines.liger_validation import audit
    from scripts.accept_liger_sampled_scaling import validate_numpy_policy
    m = contract.binding(path)
    assert m['formal_training_ready'] is True and m['phase'] == 'sampled_first_pass_formal'
    directory = Path(m['run_directory'])
    result_path = directory / 'result.json'
    r = json.loads(result_path.read_text())
    assert r['status'] == 'passed' and r['job_id'] == os.environ['SLURM_JOB_ID']
    assert r['source_git_commit'] == m['source_git_commit'] and r['manifest_sha256'] == sha(path)
    assert r['stopped_for_author_acceptance'] and not r['test_records_read'] and not r['automatic_successor']
    ctx = build_context(m, build_model=False)
    c, t = ctx.config, ctx.config['training']
    assert r['checkpoint_binding'] == ctx.binding
    assert r['initial_state_sha256'] == m['expected_initial_state_sha256']
    assert r['initial_progress'] == initial_progress(**ctx.training_budget)
    assert r['resources']['allocated_cpus'] == r['resources']['user_cpu_ceiling'] == 4
    assert r['resources']['library_threads'] == 1 and 1 <= len(r['resources']['allowed_cpus']) <= 4
    assert r['cpu_checks'] and all(check['allowed_cpus'] == r['resources']['allowed_cpus']
        and check['processes'] and all(_integer(p['threads_checked'], 1) for p in check['processes'])
        for check in r['cpu_checks'])
    assert '4090' in r['gpu']['name'] and r['gpu']['uuid']
    validate_numpy_policy(r['numpy_memory_policy'], r['versions']['numpy'])
    cache = r['feature_cache']
    assert cache['verification']['status'] == 'passed' and cache['verification']['feature_values_bitwise_equal']
    assert cache['build']['cache_sha256'] == cache['verification']['cache_sha256'] == m['expected_feature_cache_sha256']
    assert cache['build']['cache_bytes'] == cache['verification']['cache_bytes'] == 5909890560
    measured = audit_step_log(directory / 'steps.jsonl', c)
    objective = sampling_binding(c['dense_sampling'])
    p = r['final_progress']
    check_progress(p, progress_budget(c))
    assert p['epoch'] == 1 and p['next_batch'] == p['samples_in_epoch'] == 0
    assert p['batches_consumed'] == measured['attempts']
    assert p['dense_objective'] == objective and p['updates'] > 0
    for key in ('attempts', 'updates', 'skipped_updates', 'prefix_count', 'final_count', 'data_chain'):
        assert p[key] == measured[key]
    expected_batches = list(range(t['checkpoint_every_batches'], t['batches_per_epoch'] + 1,
                                  t['checkpoint_every_batches']))
    expected_names = [f'pass01-batch{batch:06d}.pt' for batch in expected_batches]
    expected_names += ['pass01-before-validation.pt', 'pass01-accepted.pt']
    assert [Path(entry['path']).name for entry in r['checkpoints']] == expected_names
    before_state = final_state = None
    for index, entry in enumerate(r['checkpoints']):
        assert Path(entry['path']).resolve() == directory / 'checkpoints' / expected_names[index]
        assert sha(entry['path']) == entry['sha256']
        state = torch.load(entry['path'], map_location='cpu', weights_only=False)
        assert state['binding'] == ctx.binding and tree_digest(state) == entry['full_state_sha256']
        sp = state['progress']
        check_progress(sp, progress_budget(c))
        assert sp['dense_objective'] == objective
        for key in ('epoch', 'next_batch', 'updates', 'attempts'):
            assert entry[key] == sp[key]
        assert state['scheduler']['total'] == t['max_optimizer_steps']
        assert state['scheduler']['warmup'] == t['warmup_steps']
        assert state['scheduler']['base_lr'] == c['optimizer']['learning_rate']
        assert state['scheduler']['updates'] == sp['updates']
        assert state['model_training'] is True and state['scaler']['scale'] > 0 and len(state['rng']['cuda']) == 1
        assert len(state['optimizer']['state']) == (152 if sp['updates'] else 0)
        assert all(int(value['step']) == sp['updates'] for value in state['optimizer']['state'].values())
        if index < len(expected_batches):
            cursor = measured['checkpoint_cursors'][expected_batches[index]]
            assert sp['epoch'] == 0 and sp['next_batch'] == expected_batches[index]
            assert sp['best'] is None and sp['validations'] == []
            assert all(sp[key] == value for key, value in cursor.items())
        elif index == len(expected_batches):
            before_state = state
            assert sp['epoch'] == 0 and sp['next_batch'] == t['batches_per_epoch']
            assert sp['best'] is None and sp['validations'] == []
            assert all(sp[key] == measured[key] for key in ('attempts', 'updates', 'skipped_updates',
                'prefix_count', 'final_count', 'data_chain'))
        else:
            final_state = state
            assert sp == p
    assert len(r['validations']) == len(p['validations']) == 1 and r['validations'] == p['validations']
    v = r['validations'][0]
    assert v['epoch'] == 1 and v['checkpoint'] == r['checkpoints'][-2]
    assert sha(v['report']['path']) == v['report']['sha256']
    assert sha(v['acceptance']['path']) == v['acceptance']['sha256']
    assert Path(v['report']['path']).resolve().parent == directory / 'validation-pass01'
    recomputed = audit(directory / 'validation-pass01', ctx.seen, ctx.index,
        c['validation']['records'], m['asset_rows'] - len(ctx.seen), v['checkpoint'], write_receipt=False)
    assert recomputed == json.loads(Path(v['acceptance']['path']).read_text())
    assert v['metrics'] == recomputed['metrics']
    proof = v['state_preservation']
    assert proof['full_learning_state_unchanged'] is True
    assert proof['before_state_sha256'] == proof['after_state_sha256'] == tree_digest(before_state)
    assert tree_digest(before_state['model']) == tree_digest(final_state['model']) == json.loads(
        Path(v['report']['path']).read_text())['model_sha256']
    for key in ('optimizer', 'scheduler', 'scaler', 'rng', 'binding', 'model_training'):
        assert tree_digest(before_state[key]) == tree_digest(final_state[key])
    assert better_validation(v['metrics']['hybrid'], 1, None)
    assert r['best'] == p['best'] == dict(epoch=1, checkpoint=v['checkpoint'], **v['metrics']['hybrid'])
    contract.binding(path)
    receipt = dict(status='passed', job_id=os.environ['SLURM_JOB_ID'], manifest_sha256=sha(path),
        source_git_commit=m['source_git_commit'], report=dict(path=str(result_path), sha256=sha(result_path)),
        full_training_parents=t['parents_per_epoch'], full_training_prefixes=t['prefixes_per_epoch'],
        full_validation_records=c['validation']['records'], completed_passes=1,
        updates=p['updates'], skipped_updates=p['skipped_updates'], best=p['best'],
        measured=measured, full_validation_state_unchanged=True, test_records_read=False,
        automatic_successor=False, requires_author_acceptance_before_next_pass=True,
        checked_at=datetime.now().astimezone().isoformat())
    write_new(directory / 'acceptance.json', receipt)
    print('[liger-sampled-train] complete one-pass training and full validation accepted; no successor', flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['run', 'accept'])
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    (run if args.mode == 'run' else accept)(args.manifest.resolve())
