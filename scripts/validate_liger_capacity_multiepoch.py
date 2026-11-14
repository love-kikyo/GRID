"""Full two-GPU validation and metadata-only single-GPU state handoff.

The evaluator never restores the training optimizer or its one-GPU RNG into
the two-GPU process. The next checkpoint is assembled on CPU from the exact
training checkpoint; only validated progress/early-stopping metadata changes.
"""
import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import time
import traceback

import torch

from scripts.probe_liger_cpu4 import sha, write_new
from scripts.validate_liger_capacity_cpu4 import load_model_checkpoint, evaluate_context
from scripts.train_liger_capacity_cpu4 import assert_finite_tensors
from src.baselines.liger_prefix_training import check_progress
from src.baselines.liger_state import save_checkpoint, tree_digest
from src.baselines.liger_early_stopping import advance_policy, validate_policy


def entry(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def _read(ref):
    assert isinstance(ref, dict) and set(ref) == {'path', 'sha256'}
    assert sha(ref['path']) == ref['sha256'], 'Bound report changed'
    return json.loads(Path(ref['path']).read_text())


def _budget(c):
    return {key: c['training'][key] for key in (
        'max_optimizer_steps', 'max_attempts', 'batch_size', 'prefixes_per_epoch')}


def check_pending_validation(p, c, epoch):
    """Exactly one complete, not-yet-validated epoch, with prior policy intact."""
    t = c['training']
    assert type(epoch) is int and 1 <= epoch <= t['max_epochs']
    check_progress(p, _budget(c))
    validate_policy(p['early_stopping'], c)
    policy = p['early_stopping']
    assert policy['completed_epochs'] == p['epoch'] == epoch - 1
    assert policy['stopped'] is False
    assert p['best'] == policy['best'] and p['validations'] == policy['validations']
    assert p['next_batch'] == t['batches_per_epoch']
    assert p['samples_in_epoch'] == t['prefixes_per_epoch']
    assert p['attempts'] == epoch * t['batches_per_epoch']
    assert p['prefix_count'] == epoch * t['prefixes_per_epoch']
    assert p['final_count'] == epoch * t['parents_per_epoch']
    assert 0 < p['updates'] <= p['attempts']


def _validation_evidence(v, c, checkpoint, *, diagnostic):
    assert v['checkpoint'] == checkpoint
    report, accepted = _read(v['report']), _read(v['acceptance'])
    assert report['status'] == accepted['status'] == 'passed'
    expected = c['smoke']['validation_records'] if diagnostic else c['validation']['records']
    assert report['records'] == accepted['records'] == expected
    assert report['checkpoint'] == accepted['checkpoint'] == checkpoint
    assert accepted['report'] == v['report']
    assert report['metrics'] == accepted['metrics'] == v['metrics']
    assert report['cold_rows'] == accepted['full_cold_rows'] == 23915712
    assert report['model_weights_unchanged'] is accepted['model_weights_unchanged'] is True
    assert report['gt_independence_verified'] is True
    assert report['test_records_read'] is accepted['test_records_read'] is False
    return report, accepted


def finalize_checkpoint(checkpoint, binding, c, validation_entry, output_path, *, diagnostic=False):
    """Return (next checkpoint entry, next progress, preservation proof).

    A diagnostic gate may use a three-batch census but retains the real global
    schedule. Formal calls require the complete authorized epoch census. No
    model, optimizer, schedule, scaler, RNG, source binding or mode is replaced
    by values from the validation runtime.
    """
    purpose = binding['purpose']
    assert type(diagnostic) is bool
    assert purpose == ('capacity384_multiepoch_gate' if diagnostic else 'capacity384_multiepoch_formal')
    if not diagnostic:
        assert (c['training']['batches_per_epoch'], c['training']['prefixes_per_epoch'],
                c['training']['parents_per_epoch'], c['validation']['records']) == (
                    22460, 45996113, 6573674, 347416)
    assert sha(checkpoint['path']) == checkpoint['sha256'], 'Checkpoint bytes changed'
    state = torch.load(checkpoint['path'], map_location='cpu', weights_only=False)
    assert tree_digest(state) == checkpoint['full_state_sha256'], 'Full training state changed'
    assert set(state) == {'format_version', 'binding', 'model', 'optimizer', 'scheduler',
                          'scaler', 'rng', 'progress', 'model_training'}
    assert state['format_version'] == 1 and state['binding'] == binding
    assert state['model_training'] is True
    assert set(state['rng']) == {'python', 'numpy', 'torch', 'cuda'} and len(state['rng']['cuda']) == 1
    assert_finite_tensors(state['model'])
    assert_finite_tensors(state['optimizer'])
    before = state['progress']
    epoch = validation_entry['epoch']
    check_pending_validation(before, c, epoch)
    for key in ('epoch', 'next_batch', 'updates', 'attempts'):
        assert checkpoint[key] == before[key]
    t = c['training']
    assert state['scheduler'] == dict(total=t['max_optimizer_steps'], warmup=t['warmup_steps'],
        base_lr=c['optimizer']['learning_rate'], updates=before['updates'])
    assert state['optimizer']['state']
    assert all(int(v['step']) == before['updates'] for v in state['optimizer']['state'].values())
    _validation_evidence(validation_entry, c, checkpoint, diagnostic=diagnostic)
    policy = advance_policy(before['early_stopping'], validation_entry, c)
    progress = copy.deepcopy(before)
    progress.update(epoch=epoch, next_batch=0, samples_in_epoch=0,
        early_stopping=policy, best=copy.deepcopy(policy['best']),
        validations=copy.deepcopy(policy['validations']))
    check_progress(progress, _budget(c))
    preserved = {key: value for key, value in state.items() if key != 'progress'}
    before_sha = tree_digest(preserved)
    state['progress'] = progress
    after_sha = tree_digest({key: value for key, value in state.items() if key != 'progress'})
    assert before_sha == after_sha
    output_path = Path(output_path).resolve()
    assert output_path != Path(checkpoint['path']).resolve()
    digest = tree_digest(state)
    save_checkpoint(output_path, state)
    next_checkpoint = dict(path=str(output_path), sha256=sha(output_path), full_state_sha256=digest,
        **{key: progress[key] for key in ('epoch', 'next_batch', 'updates', 'attempts')})
    assert sha(checkpoint['path']) == checkpoint['sha256'], 'Original training checkpoint changed'
    proof = dict(non_progress_state_unchanged=True, before_non_progress_sha256=before_sha,
        after_non_progress_sha256=after_sha, preserved_cuda_rng_devices=1,
        changed_fields=['progress'], original_checkpoint=copy.deepcopy(checkpoint),
        handoff_scope='CPU metadata-only; never capture two-GPU validation RNG or optimizer',
        diagnostic=diagnostic)
    return next_checkpoint, progress, proof


def training_receipt(path, m, c, epoch):
    assert type(epoch) is int and 1 <= epoch <= c['training']['max_epochs']
    directory = Path(m['run_directory']) / f'epoch{epoch:02d}'
    receipt_ref = entry(directory/'training_acceptance.json')
    r = _read(receipt_ref)
    assert r['status'] == 'passed_training' and r['awaiting_validation'] is True
    assert r['source_git_commit'] == m['source_git_commit'] and r['manifest_sha256'] == sha(path)
    assert r['test_records_read'] is False and r['automatic_successor'] is False
    assert r['completed_training_passes'] == epoch and r['completed_validation_passes'] == epoch - 1
    assert Path(r['report']['path']).resolve() == directory/'training_result.json'
    report = _read(r['report'])
    assert report['status'] == 'passed' and report['epoch'] == epoch
    assert report['manifest_sha256'] == sha(path) and report['source_git_commit'] == m['source_git_commit']
    assert report['checkpoint'] == r['checkpoint']
    assert report['checkpoint_binding'] == r['checkpoint_binding']
    assert report['final_progress'] == r['final_progress']
    check_pending_validation(r['final_progress'], c, epoch)
    for key, value in r['counts'].items(): assert r['final_progress'][key] == value
    checkpoint = r['checkpoint']
    # The training runner uses separate immutable segment directories on resume.
    assert Path(checkpoint['path']).resolve().is_relative_to(directory/'segments')
    assert Path(checkpoint['path']).name == 'before-validation.pt'
    assert sha(checkpoint['path']) == checkpoint['sha256']
    return r, receipt_ref


def run(path, epoch):
    from scripts import liger_capacity_multiepoch_manifest as contract
    from scripts.train_liger_capacity_multiepoch import build_context
    m, c = contract.binding(path)
    assert m['phase'] == contract.FORMAL_PHASE and m['formal_training_ready'] is True
    directory = Path(m['run_directory']) / f'epoch{epoch:02d}'
    assert directory.is_dir()
    assert not any((directory/name).exists() for name in (
        'result.json', 'acceptance.json', 'validation', 'after-validation.pt'))
    receipt, receipt_ref = training_receipt(path, m, c, epoch)
    result = dict(status='failed', phase=m['phase'], epoch=epoch, job_id=os.environ['SLURM_JOB_ID'],
        source_git_commit=m['source_git_commit'], manifest_sha256=sha(path),
        training_acceptance=receipt_ref, test_records_read=False, automatic_successor=False,
        requires_manual_next_job=True)
    started = time.monotonic()
    try:
        ctx = build_context(m, devices=2)
        assert receipt['checkpoint_binding'] == ctx.binding
        loaded = load_model_checkpoint(ctx, receipt['checkpoint'], expected_progress=receipt['final_progress'])
        report, accepted, proof = evaluate_context(ctx, directory/'validation',
            c['validation']['records'], receipt['checkpoint'])
        validation = dict(epoch=epoch, checkpoint=receipt['checkpoint'], report=accepted['report'],
            acceptance=entry(directory/'validation/acceptance.json'), metrics=report['metrics'],
            state_preservation=proof)
        contract.binding(path)
        assert sha(receipt_ref['path']) == receipt_ref['sha256']
        checkpoint, progress, handoff = finalize_checkpoint(receipt['checkpoint'], ctx.binding, c,
            validation, directory/'after-validation.pt')
        policy = progress['early_stopping']
        result.update(status='passed', completed_passes=epoch, validation=validation,
            validations=policy['validations'], best=policy['best'], policy=policy,
            stopped=policy['stopped'], stop_reason=policy['stop_reason'],
            checkpoint=checkpoint, next_checkpoint=checkpoint, next_progress=progress,
            training_progress=loaded['progress'], load_proof=loaded, handoff_proof=handoff,
            validation_model_weights_unchanged=proof['model_weights_unchanged'],
            raw_generation_coverage=report['raw_generation_coverage'],
            elapsed_seconds=time.monotonic()-started)
    except Exception:
        result['error'] = traceback.format_exc()
        raise
    finally:
        result['checked_at'] = datetime.now().astimezone().isoformat()
        write_new(directory/'result.json', result)
    write_new(directory/'acceptance.json', dict(status='passed', epoch=epoch,
        job_id=os.environ['SLURM_JOB_ID'], manifest_sha256=sha(path), source_git_commit=m['source_git_commit'],
        report=entry(directory/'result.json'), training_acceptance=receipt_ref,
        checkpoint=checkpoint, next_checkpoint=checkpoint, next_progress=progress,
        checkpoint_binding=ctx.binding, policy=policy, best=policy['best'],
        stopped=policy['stopped'], stop_reason=policy['stop_reason'], completed_passes=epoch,
        validation=validation, full_validation_records=c['validation']['records'],
        full_training_prefixes=progress['prefix_count'], full_training_parents=progress['final_count'],
        validation_model_weights_unchanged=True, handoff_proof=handoff,
        raw_generation_coverage=report['raw_generation_coverage'], test_records_read=False,
        automatic_successor=False, requires_manual_next_job=True,
        checked_at=datetime.now().astimezone().isoformat()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--epoch', type=int, required=True)
    args = parser.parse_args()
    run(args.manifest.resolve(), args.epoch)
