"""One authorized LIGER epoch, full validation, then stop for acceptance."""
import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

from scripts.liger_training_manifest import (binding, prepare, prepare_continuation,
    stage_bounds, verify_continuation, checkpoint_binding)
from scripts.probe_liger_cpu4 import sha, write_new


def restore_formal(ctx, m):
    """Restore with the predecessor's exact binding; verify every state value.

    Only newly written checkpoints use the explicitly migrated binding.
    No prior checkpoint or manifest is edited.
    """
    import torch
    from src.baselines.liger_state import capture, restore_checkpoint, tree_digest
    expected = ctx.binding
    previous = None
    if m['phase'] == 'epoch_continuation':
        old, previous = verify_continuation(m)
        expected = checkpoint_binding(old)
    path = m['resume_checkpoint']['path']
    assert sha(path) == m['resume_checkpoint']['sha256']
    saved = torch.load(path, map_location='cpu', weights_only=False)
    p = restore_checkpoint(path, *ctx.components, expected)
    restored = capture(*ctx.components, p, expected)
    assert tree_digest(restored) == tree_digest(saved), 'Resume changed full checkpoint state'
    if previous is not None: assert p == previous['final_progress']
    return p, dict(checkpoint=m['resume_checkpoint'], full_state_exact=True,
        full_state_sha256=tree_digest(saved), model_sha256=tree_digest(saved['model']),
        restored_updates=ctx.components[2].updates,
        next_learning_rate=ctx.components[1].param_groups[0]['lr'],
        original_scheduler=saved['scheduler'], binding_migration=m.get('continuation'))


def run(path):
    import torch
    from src.baselines.liger_runtime import Runtime
    from src.baselines.liger_state import capture, save_checkpoint, better_validation
    from src.baselines.liger_training import initial_progress, train_step, check_progress
    from src.baselines.liger_validation import evaluate, audit
    m = binding(path)
    assert m['formal_training_ready'] and m['phase'] in ('first_epoch_formal_training', 'epoch_continuation')
    directory = Path(m['run_directory'])
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(status='failed', phase=m['phase'], job_id=os.environ['SLURM_JOB_ID'],
                  source_git_commit=m['source_git_commit'], manifest_sha256=sha(path),
                  test_records_read=False, checkpoints=[], validations=[])
    started = time.monotonic()
    try:
        ctx = Runtime(m)
        budget = ctx.config['training']
        start_epoch, stop_epoch = stage_bounds(m, ctx.config)
        p = initial_progress()
        if m.get('resume_checkpoint'):
            p, report['resume_verification'] = restore_formal(ctx, m)
        check_progress(p, budget)
        assert p['epoch'] == start_epoch and p['epoch'] < stop_epoch
        report['stage'] = dict(start_epoch=start_epoch, stop_after_epoch=stop_epoch)
        report.update(initial_progress=copy.deepcopy(p), initialization=m['initialization'],
                      initial_state_sha256=report.get('resume_verification', {}).get('model_sha256', ctx.initial_sha),
                      resources=ctx.resources, gpu=ctx.gpu, versions=ctx.versions)
        write_new(directory / 'preflight.json', dict(report, status='ready', config=ctx.config,
            training_seen_rows=len(ctx.seen), cold_rows=len(ctx.cold), cpu_checks=ctx.cpu_checks,
            checked_at=datetime.now().astimezone().isoformat()))

        def checkpoint(name):
            target = directory / 'checkpoints' / name
            save_checkpoint(target, capture(*ctx.components, p, ctx.binding))
            entry = dict(path=str(target), sha256=sha(target), epoch=p['epoch'], next_batch=p['next_batch'], updates=p['updates'])
            report['checkpoints'].append(entry)
            print('[liger-train] checkpoint ' + json.dumps(entry), flush=True)
            return entry

        with (directory / 'steps.jsonl').open('x') as log:
            while p['epoch'] < stop_epoch:
                epoch = p['epoch']
                if p['next_batch'] < budget['batches_per_epoch']:
                    stream = ctx.batches(training=True, epoch=epoch, start_batch=p['next_batch'])
                    try:
                        while True:
                            torch.cuda.synchronize()
                            step_start = time.monotonic()
                            try: index, records = next(stream)
                            except StopIteration: break
                            assert index == p['next_batch'] < budget['batches_per_epoch']
                            expected = min(budget['batch_size'], budget['records_per_epoch'] - p['samples_in_epoch'])
                            assert len(records) == expected > 0
                            step = train_step(*ctx.components, ctx.method, ctx.official, records,
                                ctx.sids, ctx.content, ctx.seen, ctx.catalog, p, chunk_size=budget['chunk_size'])
                            torch.cuda.synchronize()
                            step.update(epoch=epoch + 1, seconds=time.monotonic() - step_start, data_chain=p['data_chain'])
                            check_progress(p, budget)
                            log.write(json.dumps(step, allow_nan=False) + '\n')
                            log.flush()
                            if p['next_batch'] == 1 or p['next_batch'] % 25 == 0:
                                ctx.cpu_check()
                                print('[liger-train] ' + json.dumps(step), flush=True)
                            if p['next_batch'] % budget['checkpoint_every_batches'] == 0:
                                checkpoint(f'epoch{epoch+1:02d}-batch{p["next_batch"]:06d}.pt')
                    finally:
                        stream.close()
                assert p['samples_in_epoch'] == budget['records_per_epoch']
                assert p['next_batch'] == budget['batches_per_epoch']
                before_val = checkpoint(f'epoch{epoch+1:02d}-before-validation.pt')
                val_dir = directory / f'validation-epoch{epoch+1:02d}'
                evaluated = evaluate(ctx.model, ctx.method, ctx.official, ctx.batches(training=False),
                    ctx.sids, ctx.content, ctx.index, ctx.seen, ctx.cold, val_dir,
                    ctx.config['validation']['records'], before_val)
                audited = audit(val_dir, ctx.seen, ctx.index, ctx.config['validation']['records'], len(ctx.cold), before_val)
                entry = dict(epoch=epoch + 1, report=audited['report'],
                             acceptance=dict(path=str(val_dir / 'acceptance.json'), sha256=sha(val_dir / 'acceptance.json')),
                             checkpoint=before_val, metrics=evaluated['metrics'])
                report['validations'].append(entry)
                if better_validation(evaluated['metrics']['hybrid'], epoch + 1, p['best']):
                    p['best'] = dict(epoch=epoch + 1, checkpoint=before_val, **evaluated['metrics']['hybrid'])
                p['validations'].append(entry)
                p.update(epoch=epoch + 1, next_batch=0, samples_in_epoch=0)
                check_progress(p, budget)
                checkpoint(f'epoch{epoch+1:02d}-accepted.pt')
                ctx.cpu_check()
        binding(path)
        report.update(status='passed', final_progress=p, best=p['best'], cpu_checks=ctx.cpu_checks,
                      elapsed_seconds=time.monotonic() - started,
                      stopped_for_author_acceptance=True, automatic_successor=False)
    except Exception:
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['checked_at'] = datetime.now().astimezone().isoformat()
        write_new(directory / 'result.json', report)


def accept(path):
    import torch
    from src.baselines.liger_runtime import Runtime
    from src.baselines.liger_state import tree_digest, better_validation
    from src.baselines.liger_training import check_progress
    from src.baselines.liger_validation import audit
    m = binding(path)
    assert m['formal_training_ready']
    directory = Path(m['run_directory'])
    report_path = directory / 'result.json'
    r = json.loads(report_path.read_text())
    assert r['status'] == 'passed' and r['job_id'] == os.environ['SLURM_JOB_ID']
    assert r['manifest_sha256'] == sha(path) and r['source_git_commit'] == m['source_git_commit']
    assert r['stopped_for_author_acceptance'] and not r['test_records_read'] and not r['automatic_successor']
    ctx = Runtime(m, build_model=False)
    assert r['resources']['allocated_cpus'] == r['resources']['user_cpu_ceiling'] == 4
    assert r['cpu_checks'] and all(c['allowed_cpus'] == r['resources']['allowed_cpus'] and c['processes'] for c in r['cpu_checks'])
    p = r['final_progress']
    check_progress(p, ctx.config['training'])
    start_epoch, stop_epoch = stage_bounds(m, ctx.config)
    assert r['initial_progress']['epoch'] == start_epoch
    assert p['epoch'] == stop_epoch and p['next_batch'] == 0 and p['batches_consumed'] == 6420 * stop_epoch
    if m['phase'] == 'epoch_continuation':
        _, previous = verify_continuation(m)
        assert r['initial_progress'] == previous['final_progress']
        assert r['resume_verification']['full_state_exact']
        source = torch.load(m['resume_checkpoint']['path'], map_location='cpu', weights_only=False)
        assert tree_digest(source) == r['resume_verification']['full_state_sha256']
    count, updates, skips, chain = (r['initial_progress'][k] for k in
                                 ('samples_in_epoch', 'updates', 'skipped_updates', 'data_chain'))
    next_batch = r['initial_progress']['next_batch']
    with (directory / 'steps.jsonl').open() as log:
        for line in log:
            step = json.loads(line)
            next_batch += 1
            assert step['batch'] == next_batch and step['epoch'] == stop_epoch
            assert step['rows'] == min(1024, 6573674 - count) and step['gradient_tensors'] == 152
            assert step['learning_rate'] >= 0 and step['seconds'] > 0
            update_number = min(updates + 1, 19260)
            scale = update_number / 963 if update_number <= 963 else .5 * (
                1 + math.cos(math.pi * (update_number - 963) / (19260 - 963)))
            assert math.isclose(step['learning_rate'], .0003 * scale, rel_tol=1e-14, abs_tol=1e-18)
            count += step['rows']
            updates += not step['skipped']
            skips += step['skipped']
            assert step['updates'] == updates
            chain = hashlib.sha256((chain + step['batch_identity_sha256']).encode()).hexdigest()
            assert chain == step['data_chain']
    assert count == 6573674 and next_batch == 6420
    assert updates == p['updates'] and skips == p['skipped_updates'] and chain == p['data_chain']
    for entry in r['checkpoints']: assert sha(entry['path']) == entry['sha256']
    assert len(r['validations']) == 1 and len(p['validations']) == stop_epoch
    assert p['validations'] == r['initial_progress']['validations'] + r['validations']
    v = r['validations'][0]
    assert sha(v['report']['path']) == v['report']['sha256']
    assert sha(v['acceptance']['path']) == v['acceptance']['sha256']
    recalculated = audit(Path(v['report']['path']).parent, ctx.seen, ctx.index, 347416, 23915712,
                         v['checkpoint'], write_receipt=False)
    assert recalculated == json.loads(Path(v['acceptance']['path']).read_text())
    assert v['metrics'] == recalculated['metrics']
    state = torch.load(r['checkpoints'][-1]['path'], map_location='cpu', weights_only=False)
    assert state['binding'] == ctx.binding and state['progress'] == p
    assert state['scheduler']['total'] == 19260 and state['scheduler']['warmup'] == 963
    assert state['scheduler']['updates'] == updates
    assert len(state['optimizer']['state']) == 152 and all(int(s['step']) == updates for s in state['optimizer']['state'].values())
    assert state['scaler']['scale'] > 0 and len(state['rng']['cuda']) == 1
    assert tree_digest(state['model']) == json.loads(Path(v['report']['path']).read_text())['model_sha256']
    best = None
    for epoch, item in enumerate(p['validations'], 1):
        assert item['epoch'] == epoch
        metrics = item['metrics']['hybrid']
        if better_validation(metrics, epoch, best):
            best = dict(epoch=epoch, checkpoint=item['checkpoint'], **metrics)
    assert r['best'] == p['best'] == best
    write_new(directory / 'acceptance.json', dict(status='passed', job_id=os.environ['SLURM_JOB_ID'],
        checked_at=datetime.now().astimezone().isoformat(), report=dict(path=str(report_path), sha256=sha(report_path)),
        full_training_records=6573674, full_validation_records=347416, completed_epochs=stop_epoch,
        updates=updates, skipped_updates=skips, best=r['best'], no_test_access=True,
        original_max_epochs=3, requires_author_acceptance_before_next_epoch=True))
    print(f'[liger-train] epoch {stop_epoch} and full validation independently accepted; no successor', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'prepare-continuation', 'run', 'accept'])
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--smoke-acceptance', type=Path)
    parser.add_argument('--resume-checkpoint', type=Path)
    parser.add_argument('--previous-manifest', type=Path)
    parser.add_argument('--previous-acceptance', type=Path)
    args = parser.parse_args()
    path = args.manifest.resolve()
    if args.mode == 'prepare':
        if args.smoke_acceptance is None: parser.error('--smoke-acceptance is required')
        prepare(path, smoke_receipt=args.smoke_acceptance, resume_checkpoint=args.resume_checkpoint)
    elif args.mode == 'prepare-continuation':
        if not args.previous_manifest or not args.previous_acceptance:
            parser.error('--previous-manifest and --previous-acceptance are required')
        prepare_continuation(path, args.previous_manifest, args.previous_acceptance)
    elif args.mode == 'run': run(path)
    else: accept(path)
