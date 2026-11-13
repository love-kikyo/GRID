"""One accepted 384-wide checkpoint, two GPUs, full validation; no training."""
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

import numpy as np
import torch

from scripts.probe_liger_cpu4 import sha, write_new
from src.baselines.liger_data import pack_batch
from src.baselines.liger_inference import generate_queries
from src.baselines.liger_retrieval import ranking_metrics
from src.baselines.liger_sharded_retrieval import ShardedColdCatalogCache
from src.baselines.liger_state import tree_digest
from src.baselines.liger_validation import PredictionWriter, add_metrics, audit


def _sync():
    for device in range(2): torch.cuda.synchronize(device)


def load_model_checkpoint(ctx, entry, *, expected_progress=None):
    """Verify the complete saved state, but restore only model weights.

    In particular, never install a one-GPU CUDA RNG snapshot in this two-GPU
    process, or mutate the complete checkpoint kept for a future training run.
    """
    assert sha(entry['path']) == entry['sha256'], 'Checkpoint bytes changed'
    state = torch.load(entry['path'], map_location='cpu', weights_only=False)
    assert tree_digest(state) == entry['full_state_sha256'], 'Checkpoint state changed'
    assert state['format_version'] == 1 and state['binding'] == ctx.binding
    assert state['model_training'] is True
    p = state['progress']
    if expected_progress is not None: assert p == expected_progress
    for key in ('epoch', 'next_batch', 'updates', 'attempts'):
        assert entry[key] == p[key]
    assert state['scheduler']['updates'] == p['updates']
    for key in ('total', 'warmup', 'base_lr'):
        assert state['scheduler'][key] == ctx.scheduler.state_dict()[key]
    assert all(torch.isfinite(value).all().item() for value in state['model'].values())
    model_sha = tree_digest(state['model'])
    ctx.model.load_state_dict(state['model'], strict=True)
    assert tree_digest(ctx.model.state_dict()) == model_sha
    ctx.model.eval()
    return dict(checkpoint=copy.deepcopy(entry), progress=copy.deepcopy(p),
                model_state_sha256=model_sha, load_scope='model_weights_only')


def _coverage(arrays):
    raw, targets = arrays['raw_beam_rows'], arrays['target_rows']
    return {str(k): int((raw[:, :k] == targets[:, None]).any(axis=1).sum())
            for k in (5, 10, 20)}


def audit_extended(directory, ctx, expected_records, checkpoint):
    """Legacy independent ranking/SID audit, plus timing and raw beam coverage."""
    directory = Path(directory)
    accepted = audit(directory, ctx.seen, ctx.index, expected_records,
                     len(ctx.cold), checkpoint, write_receipt=False)
    report = json.loads((directory / 'report.json').read_text())
    assert report['model_weights_unchanged'] is True
    assert report['shard_sizes'] == [11957856, 11957856]
    assert sum(report['shard_sizes']) == report['cold_rows'] == 23915712
    assert len(report['gpus']) == 2
    for i in range(2):
        assert 0 < report['peak_allocated_bytes'][i] <= report['peak_reserved_bytes'][i] <= report['gpus'][i]['total_bytes']
    assert report['resources']['allocated_cpus'] == 4
    assert report['resources']['library_threads'] == 1
    assert report['cpu_checks'] and all(
        value['allowed_cpus'] == report['resources']['allowed_cpus'] for value in report['cpu_checks'])
    keys = ('input_seconds', 'packing_seconds', 'generation_seconds', 'ranking_seconds',
            'serialization_metrics_seconds', 'gt_independence_seconds')
    totals, count, batches = {key: 0. for key in keys}, 0, 0
    log = directory / 'batches.jsonl'
    assert sha(log) == report['batch_log']['sha256']
    with log.open() as handle:
        for line in handle:
            step = json.loads(line)
            expected = min(32, expected_records - count)
            assert step['batch'] == batches and step['rows'] == expected > 0
            assert step['scanned_rows'] == 23915712
            for key in keys:
                assert isinstance(step[key], (int, float)) and math.isfinite(step[key]) and step[key] >= 0
                totals[key] += step[key]
            count += step['rows']
            batches += 1
    assert count == expected_records and batches == report['batches'] == math.ceil(expected_records / 32)
    for key in keys: assert math.isclose(totals[key], report['timing'][key], rel_tol=1e-12, abs_tol=1e-9)
    coverage = {str(k): 0 for k in (5, 10, 20)}
    for entry in report['artifacts']:
        with np.load(entry['path'], allow_pickle=False) as arrays:
            for key, value in _coverage(arrays).items(): coverage[key] += value
    assert coverage == report['raw_generation_coverage']['hits']
    for key, value in coverage.items():
        assert value / expected_records == report['raw_generation_coverage']['recall'][key]
    accepted.update(raw_generation_coverage=report['raw_generation_coverage'],
                    model_weights_unchanged=True, full_cold_rows=23915712)
    write_new(directory / 'acceptance.json', accepted)
    return accepted


def evaluate_context(ctx, directory, expected_records, checkpoint, *, validation_limit=None):
    """Stream the supplied validation split once, including its final short batch."""
    from src.baselines.liger_efficiency_timing import process_memory_snapshot
    assert type(expected_records) is int and expected_records > 0
    assert validation_limit is None or validation_limit == expected_records
    assert len(ctx.cold) == 23915712 and len(ctx.gpus) == 2
    directory = Path(directory).resolve()
    writer = PredictionWriter(directory, shard_rows=3200)
    started = time.monotonic()
    previous_mode = ctx.model.training
    before = tree_digest(ctx.model.state_dict())
    ctx.model.eval()
    stream = iter(ctx.batches(training=False, validation_limit=validation_limit))
    cache = None
    totals, users, identity, coverage = {}, set(), hashlib.sha256(), {str(k): 0 for k in (5, 10, 20)}
    count = batch_count = valid_beams = 0
    timing = {key: 0. for key in ('input_seconds', 'packing_seconds', 'generation_seconds',
        'ranking_seconds', 'serialization_metrics_seconds', 'gt_independence_seconds')}
    try:
        ctx.model.predicted_embedding = None
        for i in range(2):
            with torch.cuda.device(i): torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(i)
        _sync()
        begin = time.monotonic()
        cache = ShardedColdCatalogCache(ctx.model.emb_proj, ctx.content, ctx.cold, chunk_size=65536)
        _sync()
        cache_seconds = time.monotonic() - begin
        assert list(cache.shard_sizes) == [11957856, 11957856]
        with (directory / 'batches.jsonl').open('x') as log:
            while True:
                begin = time.monotonic()
                try: index, records = next(stream)
                except StopIteration: break
                step = dict(batch=batch_count, rows=len(records), input_seconds=time.monotonic()-begin)
                assert index == batch_count and len(records) == min(32, expected_records-count) > 0
                begin = time.monotonic()
                inputs, meta = pack_batch(records, ctx.sids, ctx.content)
                step['packing_seconds'] = time.monotonic()-begin
                current = set(meta['user_ids'].tolist())
                assert len(current) == len(records) and not users.intersection(current), 'Duplicate validation user'
                users.update(current)
                _sync()
                begin = time.monotonic()
                query, generation = generate_queries(ctx.model, inputs, ctx.method, ctx.official, ctx.index, 'cuda:0')
                _sync()
                step['generation_seconds'] = time.monotonic()-begin
                step['gt_independence_seconds'] = 0.
                if batch_count == 0:
                    begin = time.monotonic()
                    changed = dict(inputs, labels_sids=np.full_like(inputs['labels_sids'], 123))
                    other_q, other = generate_queries(ctx.model, changed, ctx.method, ctx.official, ctx.index, 'cuda:0')
                    torch.testing.assert_close(query, other_q, atol=0, rtol=0)
                    for key in generation: np.testing.assert_array_equal(generation[key], other[key])
                    del other_q, other
                    _sync()
                    step['gt_independence_seconds'] = time.monotonic()-begin
                begin = time.monotonic()
                ranked = cache.rank(query, generation['valid_beam_rows'])
                _sync()
                step['ranking_seconds'] = time.monotonic()-begin
                assert cache.last_scan_rows == 23915712, 'Incomplete cold scan'
                step['scanned_rows'] = cache.last_scan_rows
                begin = time.monotonic()
                arrays = dict(user_ids=meta['user_ids'], target_rows=meta['target_rows'],
                    record_sha256=np.asarray(meta['serialized_sha256'], dtype='S64'), **generation,
                    **{key: value for key, value in ranked.items() if key != 'valid_beam_rows'})
                identity.update(arrays['record_sha256'].tobytes())
                writer.append(arrays)
                add_metrics(totals, {mode: ranking_metrics(arrays[f'{mode}_rows'], meta['target_rows'])
                    for mode in ('hybrid', 'beam_only')}, len(records))
                for key, value in _coverage(arrays).items(): coverage[key] += value
                valid_beams += int((arrays['valid_beam_rows'] >= 0).sum())
                step['serialization_metrics_seconds'] = time.monotonic()-begin
                for key in timing: timing[key] += step[key]
                log.write(json.dumps(step, allow_nan=False)+'\n')
                log.flush()
                count += len(records)
                batch_count += 1
                ctx.model.predicted_embedding = None
                del query, generation, ranked, arrays, inputs
                if batch_count == 1 or batch_count % 100 == 0:
                    ctx.cpu_check()
                    print('[liger-capacity-validation] '+json.dumps(dict(records=count,
                        expected_records=expected_records, **step)), flush=True)
        begin = time.monotonic()
        writer.flush()
        final_flush_seconds = time.monotonic()-begin
        assert count == len(users) == expected_records
        after = tree_digest(ctx.model.state_dict())
        assert before == after, 'Validation changed model weights'
        ctx.cpu_check()
        report = dict(status='passed', records=count, batches=batch_count, checkpoint=checkpoint,
            model_sha256=before, record_identity_sha256=identity.hexdigest(),
            cold_rows=23915712, shard_sizes=list(cache.shard_sizes), cache_bytes=list(cache.cache_bytes),
            valid_generated_slots=valid_beams, gt_independence_verified=True,
            model_weights_unchanged=True, test_records_read=False,
            cache_seconds=cache_seconds, elapsed_seconds=time.monotonic()-started,
            peak_allocated_bytes=[torch.cuda.max_memory_allocated(i) for i in range(2)],
            peak_reserved_bytes=[torch.cuda.max_memory_reserved(i) for i in range(2)],
            gpus=ctx.gpus, resources=ctx.resources, cpu_checks=copy.deepcopy(ctx.cpu_checks),
            memory=process_memory_snapshot(), artifacts=writer.artifacts,
            batch_log=dict(path=str(directory/'batches.jsonl'), sha256=sha(directory/'batches.jsonl')),
            timing=dict(timing, final_flush_seconds=final_flush_seconds,
                scope='actual streamed input, packing, generation, full-cold ranking and output; cache and final audit separate'),
            raw_generation_coverage=dict(hits=coverage, recall={key:value/count for key,value in coverage.items()},
                meaning='GT present in the first k raw generation slots, before dense reordering'),
            metrics={mode:{key:value/count for key,value in values.items()} for mode,values in totals.items()})
        write_new(directory/'report.json', report)
        begin = time.monotonic()
        accepted = audit_extended(directory, ctx, expected_records, checkpoint)
        proof = dict(model_weights_unchanged=True, before_model_sha256=before,
            after_model_sha256=after, load_scope='model_weights_only',
            audit_seconds=time.monotonic()-begin, total_seconds=time.monotonic()-started)
        write_new(directory/'completion.json', dict(proof, checked_at=datetime.now().astimezone().isoformat()))
        return report, accepted, proof
    except Exception:
        write_new(directory/'failure.json', dict(status='failed', records=count, checkpoint=checkpoint,
            error=traceback.format_exc(), checked_at=datetime.now().astimezone().isoformat()))
        raise
    finally:
        if hasattr(stream, 'close'): stream.close()
        if cache is not None: cache.close()
        cache = None
        ctx.model.predicted_embedding = None
        ctx.model.train(previous_mode)
        for i in range(2):
            with torch.cuda.device(i): torch.cuda.empty_cache()


def _training_receipt(path, m, c):
    directory = Path(m['run_directory'])
    receipt_path = directory/'training_acceptance.json'
    r = json.loads(receipt_path.read_text())
    assert r['status'] == 'passed_training' and r['awaiting_validation'] is True
    assert r['source_git_commit'] == m['source_git_commit'] and r['manifest_sha256'] == sha(path)
    assert r['test_records_read'] is False and r['automatic_successor'] is False
    assert r['initial_state_sha256'] == m['expected_initial_state_sha256']
    for key, name in (('report', 'training_result.json'), ('steps', 'steps.jsonl')):
        assert Path(r[key]['path']).resolve() == directory/name
        assert sha(r[key]['path']) == r[key]['sha256']
    t, counts = c['training'], r['counts']
    assert counts['attempts'] == t['batches_per_epoch'] == 22460
    assert counts['prefix_count'] == t['prefixes_per_epoch'] == 45996113
    assert counts['final_count'] == t['parents_per_epoch'] == 6573674
    assert 0 < counts['updates'] <= counts['attempts']
    assert counts['updates'] + counts['skipped_updates'] == counts['attempts']
    entry = r['checkpoint']
    assert Path(entry['path']).resolve() == directory/'checkpoints/pass01-before-validation.pt'
    assert sha(entry['path']) == entry['sha256']
    return r, dict(path=str(receipt_path), sha256=sha(receipt_path))


def run(path):
    from scripts import liger_capacity_training_manifest as contract
    from scripts.train_liger_capacity_cpu4 import build_context
    m, c = contract.binding(path)
    assert m['phase'] == contract.FORMAL_PHASE and m['formal_training_ready'] is True
    directory = Path(m['run_directory'])
    assert directory.is_dir() and not any((directory/name).exists() for name in ('result.json', 'acceptance.json', 'validation-pass01'))
    receipt, receipt_entry = _training_receipt(path, m, c)
    result = dict(status='failed', job_id=os.environ['SLURM_JOB_ID'], phase=m['phase'],
        source_git_commit=m['source_git_commit'], manifest_sha256=sha(path),
        training_acceptance=receipt_entry, test_records_read=False, automatic_successor=False)
    started = time.monotonic()
    try:
        ctx = build_context(m, devices=2)
        assert receipt['checkpoint_binding'] == ctx.binding
        loaded = load_model_checkpoint(ctx, receipt['checkpoint'])
        p = loaded['progress']
        assert p['epoch'] == 0 and p['next_batch'] == p['batches_consumed'] == 22460
        assert p['samples_in_epoch'] == p['prefix_count'] == 45996113
        assert p['best'] is None and p['validations'] == []
        for key, value in receipt['counts'].items(): assert p[key] == value
        report, accepted, proof = evaluate_context(ctx, directory/'validation-pass01',
            c['validation']['records'], receipt['checkpoint'])
        contract.binding(path)
        assert sha(receipt['checkpoint']['path']) == receipt['checkpoint']['sha256']
        assert sha(receipt_entry['path']) == receipt_entry['sha256']
        entry = dict(epoch=1, checkpoint=receipt['checkpoint'], report=accepted['report'],
            acceptance=dict(path=str(directory/'validation-pass01/acceptance.json'),
                            sha256=sha(directory/'validation-pass01/acceptance.json')),
            metrics=report['metrics'], state_preservation=proof)
        result.update(status='passed', completed_passes=1, validations=[entry],
            best=dict(epoch=1, checkpoint=receipt['checkpoint'], **report['metrics']['hybrid']),
            selection_scope='single authorized pass endpoint; no multi-epoch selection',
            training_progress=loaded['progress'], load_proof=loaded,
            stopped_for_author_acceptance=True, elapsed_seconds=time.monotonic()-started)
    except Exception:
        result['error'] = traceback.format_exc()
        raise
    finally:
        result['checked_at'] = datetime.now().astimezone().isoformat()
        write_new(directory/'result.json', result)
    write_new(directory/'acceptance.json', dict(status='passed', job_id=os.environ['SLURM_JOB_ID'],
        manifest_sha256=sha(path), source_git_commit=m['source_git_commit'],
        report=dict(path=str(directory/'result.json'), sha256=sha(directory/'result.json')),
        training_acceptance=receipt_entry, full_training_prefixes=45996113,
        full_training_parents=6573674, full_validation_records=347416, completed_passes=1,
        best=result['best'], raw_generation_coverage=report['raw_generation_coverage'],
        validation_model_weights_unchanged=True, test_records_read=False, automatic_successor=False,
        requires_author_acceptance_before_next_pass=True,
        checked_at=datetime.now().astimezone().isoformat()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    run(parser.parse_args().manifest.resolve())
