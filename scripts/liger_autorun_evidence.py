"""Read-only acceptance gate for the external LIGER serial controller.

The frozen training/evaluation programs still do not submit successor jobs.
This adapter checks their existing receipts without changing that historical
fact, loading checkpoint tensors, writing artifacts, or contacting Slurm.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re

from src.baselines.liger_early_stopping import advance_policy, validate_policy

if not __debug__:
    raise RuntimeError('Acceptance gates require Python assertions; optimized mode is forbidden')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def reference(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def check_reference(ref, expected=None, *, checkpoint=False):
    assert isinstance(ref, dict)
    if not checkpoint:
        assert set(ref) == {'path', 'sha256'}
    assert isinstance(ref['path'], str) and Path(ref['path']).is_absolute()
    if expected is not None:
        assert Path(ref['path']).resolve() == Path(expected).resolve(), 'Unexpected artifact path'
    assert isinstance(ref['sha256'], str) and re.fullmatch('[0-9a-f]{64}', ref['sha256'])
    assert sha(ref['path']) == ref['sha256'], 'Artifact bytes changed'
    if checkpoint:
        assert re.fullmatch('[0-9a-f]{64}', ref['full_state_sha256'])
    return ref


def read_bound(ref, expected=None):
    check_reference(ref, expected)
    return json.loads(Path(ref['path']).read_text())


def require_authority(config):
    t, e = config['training'], config['early_stopping']
    assert (t['min_epochs'], t['max_epochs'], e['min_epochs'], e['max_epochs'], e['patience']) == (6, 12, 6, 12, 5)
    assert (t['batches_per_epoch'], t['prefixes_per_epoch'], t['parents_per_epoch']) == (22460, 45996113, 6573674)
    assert t['batch_size'] == 2048 and t['final_batch_size'] == 81
    assert t['max_attempts'] == t['max_optimizer_steps'] == 269520
    assert config['validation']['records'] == 347416 and config['validation']['batch_size'] == 32
    assert config['validation']['cold_rows'] == 23915712


def _identity(value, *, epoch, job, manifest_sha, source, status):
    assert value['status'] == status and value['epoch'] == epoch
    assert value['job_id'] == job, 'Receipt belongs to a different Slurm job'
    assert value['manifest_sha256'] == manifest_sha and value['source_git_commit'] == source
    assert value['test_records_read'] is False and value['automatic_successor'] is False


def _progress_census(progress, config, epoch):
    t = config['training']
    for key in ('epoch', 'next_batch', 'samples_in_epoch', 'batches_consumed', 'attempts',
                'prefix_count', 'final_count', 'updates', 'skipped_updates',
                'max_optimizer_steps', 'max_attempts'):
        assert type(progress[key]) is int and progress[key] >= 0
    assert progress['batches_consumed'] == progress['attempts'] == epoch * t['batches_per_epoch']
    assert progress['prefix_count'] == epoch * t['prefixes_per_epoch']
    assert progress['final_count'] == epoch * t['parents_per_epoch']
    assert progress['updates'] > 0 and progress['updates'] + progress['skipped_updates'] == progress['attempts']
    assert progress['max_optimizer_steps'] == t['max_optimizer_steps']
    assert progress['max_attempts'] == t['max_attempts']
    assert re.fullmatch('[0-9a-f]{64}', progress['data_chain'])
    validate_policy(progress['early_stopping'], config)
    assert progress['best'] == progress['early_stopping']['best']
    assert progress['validations'] == progress['early_stopping']['validations']


def _checkpoint_census(checkpoint, progress):
    for key in ('epoch', 'next_batch', 'updates', 'attempts'):
        assert type(checkpoint[key]) is int and checkpoint[key] == progress[key]


def check_transition(before, after, policy, validation, handoff, checkpoint, config, epoch):
    """Check the policy and metadata transition without deserializing tensors."""
    require_authority(config)
    _progress_census(before, config, epoch)
    _progress_census(after, config, epoch)
    t = config['training']
    assert before['epoch'] == epoch - 1
    assert before['next_batch'] == t['batches_per_epoch']
    assert before['samples_in_epoch'] == t['prefixes_per_epoch']
    assert before['early_stopping']['completed_epochs'] == epoch - 1
    assert before['early_stopping']['stopped'] is False
    assert validation['epoch'] == epoch and validation['checkpoint'] == checkpoint
    expected_policy = advance_policy(before['early_stopping'], validation, config)
    assert policy == expected_policy, 'Policy was not advanced from the preceding complete state'
    expected = copy.deepcopy(before)
    expected.update(epoch=epoch, next_batch=0, samples_in_epoch=0,
                    early_stopping=expected_policy, best=expected_policy['best'],
                    validations=expected_policy['validations'])
    assert after == expected, 'Validation modified unauthorized training progress'
    assert handoff['non_progress_state_unchanged'] is True
    assert handoff['before_non_progress_sha256'] == handoff['after_non_progress_sha256']
    assert re.fullmatch('[0-9a-f]{64}', handoff['before_non_progress_sha256'])
    assert handoff['changed_fields'] == ['progress'] and handoff['preserved_cuda_rng_devices'] == 1
    assert handoff['diagnostic'] is False and handoff['original_checkpoint'] == checkpoint
    assert policy['completed_epochs'] == epoch
    if epoch == 12:
        assert policy['stopped'] is True and policy['stop_reason'] == 'max_epochs'
    if policy['stopped']:
        assert epoch >= 6 and policy['stop_reason'] in ('patience', 'max_epochs')
    return expected_policy


def audit_prediction_artifacts(report, directory, *, expected_records, batch_size,
                               cold_rows, shard_rows=3200, asset_rows=35458467):
    """Independently recompute persisted ranks and identities on one CPU.

    Full SID/candidate mapping checks already ran in the successful validation
    job. This lightweight repetition deliberately does not build that index.
    """
    import numpy as np
    directory = Path(directory).resolve()
    assert report['records'] == expected_records and report['cold_rows'] == cold_rows
    assert report['batches'] == math.ceil(expected_records / batch_size)
    assert len(report['artifacts']) == math.ceil(expected_records / shard_rows)
    totals = {mode: {f'{metric}@{k}': 0. for metric in ('Recall', 'NDCG') for k in (5, 10, 20)}
              for mode in ('hybrid', 'beam_only')}
    users, identity, count, valid_beams = set(), hashlib.sha256(), 0, 0
    coverage = {str(k): 0 for k in (5, 10, 20)}
    for index, ref in enumerate(report['artifacts']):
        assert set(ref) == {'path', 'sha256', 'rows'}
        path = directory / f'part-{index:05d}.npz'
        check_reference({k: ref[k] for k in ('path', 'sha256')}, path)
        with np.load(path, allow_pickle=False) as a:
            n = min(shard_rows, expected_records - count)
            assert ref['rows'] == n > 0
            targets, user_ids = a['target_rows'], a['user_ids']
            assert targets.shape == user_ids.shape == (n,)
            assert np.issubdtype(targets.dtype, np.integer) and np.issubdtype(user_ids.dtype, np.integer)
            assert ((targets >= 0) & (targets < asset_rows)).all()
            current = set(user_ids.tolist())
            assert len(current) == n and not users & current, 'Duplicate validation users'
            users.update(current)
            record_sha = a['record_sha256']
            assert record_sha.shape == (n,) and record_sha.dtype == np.dtype('S64')
            assert all(re.fullmatch(b'[0-9a-f]{64}', x) for x in record_sha)
            identity.update(record_sha.tobytes())
            raw, valid = a['raw_beam_rows'], a['valid_beam_rows']
            assert raw.shape == valid.shape == (n, 20)
            for rows in (raw, valid):
                assert np.issubdtype(rows.dtype, np.integer) and ((rows >= -1) & (rows < asset_rows)).all()
            valid_beams += int((valid >= 0).sum())
            for k in (5, 10, 20):
                coverage[str(k)] += int((raw[:, :k] == targets[:, None]).any(axis=1).sum())
            for mode in totals:
                rows, scores = a[f'{mode}_rows'], a[f'{mode}_scores']
                assert rows.shape == scores.shape == (n, 20)
                assert np.issubdtype(rows.dtype, np.integer) and ((rows >= -1) & (rows < asset_rows)).all()
                assert np.isfinite(scores[rows >= 0]).all() and np.isneginf(scores[rows < 0]).all()
                assert (scores[:, 1:] <= scores[:, :-1]).all()
                ordered = np.sort(rows, axis=1)
                assert not ((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any()
                if mode == 'hybrid':
                    assert (rows >= 0).all()
                hit = rows == targets[:, None]
                rank = np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, 0)
                for k in (5, 10, 20):
                    matched = (rank > 0) & (rank <= k)
                    totals[mode][f'Recall@{k}'] += int(matched.sum())
                    totals[mode][f'NDCG@{k}'] += float((1 / np.log2(rank[matched] + 1)).sum())
            count += n
    assert count == len(users) == expected_records
    assert identity.hexdigest() == report['record_identity_sha256']
    assert valid_beams == report['valid_generated_slots']
    assert set(report['metrics']) == set(totals)
    for mode, values in totals.items():
        assert set(report['metrics'][mode]) == set(values)
        for key, value in values.items():
            recorded = report['metrics'][mode][key]
            assert type(recorded) in (int, float) and math.isfinite(recorded)
            assert math.isclose(value / count, recorded, rel_tol=0, abs_tol=1e-12), 'Metric arithmetic mismatch'
    assert coverage == report['raw_generation_coverage']['hits']
    assert report['raw_generation_coverage']['recall'] == {key: value / count for key, value in coverage.items()}
    check_reference(report['batch_log'], directory/'batches.jsonl')
    timings = {key: 0. for key in ('input_seconds', 'packing_seconds', 'generation_seconds',
                                  'ranking_seconds', 'serialization_metrics_seconds', 'gt_independence_seconds')}
    batch_count = rows_count = 0
    with (directory/'batches.jsonl').open() as handle:
        for line in handle:
            value = json.loads(line)
            expected = min(batch_size, expected_records - rows_count)
            assert value['batch'] == batch_count and value['rows'] == expected > 0
            assert value['scanned_rows'] == cold_rows
            for key in timings:
                number = value[key]
                assert type(number) in (float, int) and math.isfinite(number) and number >= 0
                timings[key] += number
            rows_count += value['rows']
            batch_count += 1
    assert rows_count == count and batch_count == report['batches']
    for key, value in timings.items():
        assert math.isclose(value, report['timing'][key], rel_tol=1e-12, abs_tol=1e-9)
    return dict(records=count, artifacts=len(report['artifacts']), batches=batch_count,
                final_batch=expected_records - (batch_count - 1) * batch_size,
                cold_rows=cold_rows, metrics_recomputed=True, identities_recomputed=True,
                candidate_mapping_rechecked=False)


def verify_bound_epoch(manifest_path, manifest, config, checkpoint_binding,
                       epoch, train_job, validation_job):
    """The public verifier after the frozen manifest binding has succeeded."""
    require_authority(config)
    assert type(epoch) is int and 1 <= epoch <= 12
    assert all(isinstance(job, str) and re.fullmatch('[1-9][0-9]*', job)
               for job in (train_job, validation_job))
    assert train_job != validation_job
    assert manifest['phase'] == 'capacity384_multiepoch_formal' and manifest['formal_training_ready'] is True
    directory = Path(manifest['run_directory']).resolve()/f'epoch{epoch:02d}'
    manifest_sha, source = sha(manifest_path), manifest['source_git_commit']
    train_ref = reference(directory/'training_acceptance.json')
    train = read_bound(train_ref)
    train_report = read_bound(train['report'], directory/'training_result.json')
    _identity(train, epoch=epoch, job=train_job, manifest_sha=manifest_sha, source=source, status='passed_training')
    _identity(train_report, epoch=epoch, job=train_job, manifest_sha=manifest_sha, source=source, status='passed')
    assert train['awaiting_validation'] is train_report['awaiting_validation'] is True
    for value in (train, train_report):
        assert value['completed_training_passes'] == epoch and value['completed_validation_passes'] == epoch - 1
        assert value['checkpoint_binding'] == checkpoint_binding
        assert value['initial_state_sha256'] == manifest['expected_initial_state_sha256']
    before, checkpoint = train['final_progress'], train['checkpoint']
    assert train_report['final_progress'] == before and train_report['checkpoint'] == checkpoint
    assert train['measured']['progress'] == before and train['measured']['steps'] == config['training']['batches_per_epoch']
    assert train['segments'] == train['measured']['segments'] and train['segments']
    for key, value in train['counts'].items():
        assert before[key] == value
    assert set(train['counts']) == {'attempts', 'updates', 'skipped_updates', 'prefix_count', 'final_count', 'data_chain'}
    # Log hashes bind the node-side complete tensor/log audit, without reloading
    # all checkpoint tensors on the login node.
    for segment in train['segments']:
        segment_path = Path(segment['segment']).resolve()
        assert segment_path.parent == directory/'segments'
        read_bound(segment['preflight'], segment_path/'preflight.json')
        check_reference(segment['steps'], segment_path/'steps.jsonl')
    checkpoint_path = Path(checkpoint['path']).resolve()
    assert checkpoint_path.name == 'before-validation.pt' and checkpoint_path.is_relative_to(directory/'segments')
    check_reference(checkpoint, checkpoint=True)
    _checkpoint_census(checkpoint, before)
    acceptance_ref = reference(directory/'acceptance.json')
    accepted = read_bound(acceptance_ref)
    result = read_bound(accepted['report'], directory/'result.json')
    for value in (accepted, result):
        _identity(value, epoch=epoch, job=validation_job, manifest_sha=manifest_sha, source=source, status='passed')
        assert value['requires_manual_next_job'] is True  # frozen evaluator has no self-submission
        assert value['training_acceptance'] == train_ref and value['completed_passes'] == epoch
        assert value['validation_model_weights_unchanged'] is True
    assert result['phase'] == manifest['phase']
    assert accepted['checkpoint_binding'] == checkpoint_binding
    assert accepted['full_validation_records'] == config['validation']['records']
    assert accepted['full_training_prefixes'] == epoch * config['training']['prefixes_per_epoch']
    assert accepted['full_training_parents'] == epoch * config['training']['parents_per_epoch']
    shared = ('checkpoint', 'next_checkpoint', 'next_progress', 'policy', 'best', 'stopped',
              'stop_reason', 'validation', 'handoff_proof', 'raw_generation_coverage')
    assert all(accepted[key] == result[key] for key in shared)
    assert accepted['checkpoint'] == accepted['next_checkpoint']
    next_cp = check_reference(accepted['next_checkpoint'], directory/'after-validation.pt', checkpoint=True)
    after, validation, policy = accepted['next_progress'], accepted['validation'], accepted['policy']
    _checkpoint_census(next_cp, after)
    assert result['training_progress'] == before and result['validations'] == policy['validations']
    assert result['load_proof']['checkpoint'] == checkpoint and result['load_proof']['progress'] == before
    assert result['load_proof']['load_scope'] == 'model_weights_only'
    check_transition(before, after, policy, validation, accepted['handoff_proof'], checkpoint, config, epoch)
    assert accepted['best'] == policy['best'] and accepted['stopped'] is policy['stopped']
    assert accepted['stop_reason'] == policy['stop_reason']
    report = None
    for number, value in enumerate(policy['validations'], 1):
        epoch_dir = directory.parent/f'epoch{number:02d}'
        r = read_bound(value['report'], epoch_dir/'validation/report.json')
        receipt = read_bound(value['acceptance'], epoch_dir/'validation/acceptance.json')
        assert r['status'] == receipt['status'] == 'passed'
        assert receipt['report'] == value['report']
        assert r['records'] == receipt['records'] == config['validation']['records']
        assert r['checkpoint'] == receipt['checkpoint'] == value['checkpoint']
        assert r['metrics'] == receipt['metrics'] == value['metrics']
        assert r['cold_rows'] == receipt['full_cold_rows'] == config['validation']['cold_rows']
        assert r['test_records_read'] is receipt['test_records_read'] is False
        assert r['model_weights_unchanged'] is receipt['model_weights_unchanged'] is True
        assert r['gt_independence_verified'] is True
        if number == epoch:
            report = r
            assert receipt['raw_generation_coverage'] == r['raw_generation_coverage'] == accepted['raw_generation_coverage']
    assert policy['validations'][-1] == validation and report is not None
    assert report['model_sha256'] == result['load_proof']['model_state_sha256']
    preservation = validation['state_preservation']
    assert preservation['model_weights_unchanged'] is True and preservation['load_scope'] == 'model_weights_only'
    assert preservation['before_model_sha256'] == preservation['after_model_sha256'] == report['model_sha256']
    assert report['resources']['job_id'] == validation_job
    assert report['resources']['allocated_cpus'] == 4 and report['resources']['library_threads'] == 1
    assert len(report['gpus']) == 2 and all('4090' in gpu['name'] for gpu in report['gpus'])
    assert report['shard_sizes'] == [11957856, 11957856]
    evidence = audit_prediction_artifacts(report, directory/'validation',
        expected_records=config['validation']['records'], batch_size=config['validation']['batch_size'],
        cold_rows=config['validation']['cold_rows'], shard_rows=config['validation']['shard_rows'])
    return dict(status='accepted', epoch=epoch, stopped=policy['stopped'], stop_reason=policy['stop_reason'],
                next_epoch=None if policy['stopped'] else epoch + 1, policy=policy,
                best=policy['best'], checkpoint=next_cp, acceptance=acceptance_ref,
                train_job=train_job, validation_job=validation_job, manifest_sha256=manifest_sha,
                source_git_commit=source, artifact_audit=evidence,
                checkpoint_tensors_reloaded=False, external_controller_authority_required=True)


def verify_epoch(manifest_path, epoch, train_job, validation_job):
    # Imported lazily; importing this module has no runtime or Slurm effects.
    from scripts import liger_capacity_multiepoch_manifest as contract
    manifest_path = Path(manifest_path).resolve()
    manifest, config = contract.binding(manifest_path)
    return verify_bound_epoch(manifest_path, manifest, config, contract.checkpoint_binding(manifest),
                              epoch, train_job, validation_job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--train-job', required=True)
    parser.add_argument('--validation-job', required=True)
    args = parser.parse_args()
    print(json.dumps(verify_epoch(args.manifest, args.epoch, args.train_job, args.validation_job),
                     allow_nan=False, sort_keys=True))


if __name__ == '__main__':
    main()
