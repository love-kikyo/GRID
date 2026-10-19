"""Stream all validation records to create-only, independently auditable shards."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.probe_liger_cpu4 import sha, write_new
from src.baselines.liger_candidates import unique_beam_rows
from src.baselines.liger_data import pack_batch
from src.baselines.liger_eval_audit import audit_predictions
from src.baselines.liger_inference import generate_queries
from src.baselines.liger_retrieval import ColdCatalogCache, full_cold_rank, ranking_metrics
from src.baselines.liger_state import rng_state, restore_rng, tree_digest


def add_metrics(totals, metrics, count):
    for mode, values in metrics.items():
        for key, value in values.items():
            totals.setdefault(mode, {}).setdefault(key, 0.)
            totals[mode][key] += count * value


class PredictionWriter:
    def __init__(self, directory, shard_rows=3200):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.limit, self.pending, self.count, self.artifacts = shard_rows, [], 0, []

    def append(self, arrays):
        rows = len(arrays['user_ids'])
        if not rows or any(len(value) != rows for value in arrays.values()):
            raise ValueError('Prediction fields disagree on batch length')
        self.pending.append(arrays)
        self.count += rows
        if self.count >= self.limit: self.flush()

    def flush(self):
        if not self.pending: return
        keys = self.pending[0].keys()
        arrays = {key: np.concatenate([p[key] for p in self.pending]) for key in keys}
        path = self.directory / f'part-{len(self.artifacts):05d}.npz'
        with path.open('xb') as handle: np.savez(handle, **arrays)
        self.artifacts.append(dict(path=str(path.resolve()), sha256=sha(path), rows=self.count))
        self.pending, self.count = [], 0


def evaluate(model, method, official, batches, sids, content, index, seen, cold,
             directory, expected_records, checkpoint, *, chunk_size=65536):
    """No test access; caller explicitly supplies the frozen validation stream."""
    started = time.monotonic()
    device = next(model.parameters()).device
    before_rng, training = rng_state(), model.training
    state_hash = tree_digest(model.state_dict())
    model.eval()
    writer = PredictionWriter(directory)
    totals, users, identity = {}, set(), hashlib.sha256()
    count, batch_count, valid_beams = 0, 0, 0
    cache = None
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        cache_start = time.monotonic()
        cache = ColdCatalogCache(model.emb_proj, content, cold, device=device, chunk_size=chunk_size)
        torch.cuda.synchronize()
        cache_seconds = time.monotonic() - cache_start
        for _, records in batches:
            inputs, meta = pack_batch(records, sids, content)
            current_users = set(meta['user_ids'].tolist())
            if users & current_users: raise ValueError('Duplicate validation user')
            users.update(current_users)
            query, generation = generate_queries(model, inputs, method, official, index, device)
            if batch_count == 0:
                changed = dict(inputs, labels_sids=np.full_like(inputs['labels_sids'], 123))
                other_q, other = generate_queries(model, changed, method, official, index, device)
                torch.testing.assert_close(query, other_q, atol=0, rtol=0)
                np.testing.assert_array_equal(generation['beam_tokens'], other['beam_tokens'])
                del other_q
            ranked = full_cold_rank(cache, query, generation['valid_beam_rows'])
            if cache.last_scan_rows != len(cold): raise ValueError('Incomplete cold scan')
            arrays = dict(user_ids=meta['user_ids'], target_rows=meta['target_rows'],
                record_sha256=np.asarray(meta['serialized_sha256'], dtype='S64'), **generation,
                **{k: v for k, v in ranked.items() if k != 'valid_beam_rows'})
            identity.update(arrays['record_sha256'].tobytes())
            writer.append(arrays)
            metrics = {mode: ranking_metrics(arrays[f'{mode}_rows'], meta['target_rows'])
                       for mode in ('hybrid', 'beam_only')}
            add_metrics(totals, metrics, len(records))
            count += len(records)
            batch_count += 1
            valid_beams += int((generation['valid_beam_rows'] >= 0).sum())
            if batch_count == 1 or batch_count % 100 == 0:
                print(f'[liger-val] {count}/{expected_records} records; cold={cache.last_scan_rows}', flush=True)
        writer.flush()
        if count != expected_records or len(users) != count:
            raise ValueError(f'Validation expected {expected_records}, read {count}')
        if tree_digest(model.state_dict()) != state_hash:
            raise ValueError('Validation changed model weights')
        report = dict(status='passed', records=count, batches=batch_count, checkpoint=checkpoint,
            model_sha256=state_hash, record_identity_sha256=identity.hexdigest(),
            cold_rows=len(cold), valid_generated_slots=valid_beams,
            gt_independence_verified=True, test_records_read=False,
            cache_seconds=cache_seconds, elapsed_seconds=time.monotonic() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), artifacts=writer.artifacts,
            metrics={mode: {k: v / count for k, v in values.items()} for mode, values in totals.items()})
        write_new(Path(directory) / 'report.json', report)
        return report
    finally:
        if hasattr(batches, 'close'): batches.close()
        del cache
        model.train(training)
        restore_rng(before_rng)
        torch.cuda.empty_cache()


def audit(directory, seen, index, expected_records, expected_cold, expected_checkpoint, *, write_receipt=True):
    """Independent CPU metric arithmetic and serialized candidate/identity audit."""
    directory = Path(directory).resolve()
    report_path = directory / 'report.json'
    r = json.loads(report_path.read_text())
    assert r['status'] == 'passed' and r['records'] == expected_records
    assert r['cold_rows'] == expected_cold and r['checkpoint'] == expected_checkpoint
    assert r['gt_independence_verified'] and not r['test_records_read']
    assert r['cache_seconds'] > 0 and r['elapsed_seconds'] >= r['cache_seconds']
    assert r['artifacts']
    totals, users, identity = {}, set(), hashlib.sha256()
    count, valid_beams = 0, 0
    for i, entry in enumerate(r['artifacts']):
        path = directory / f'part-{i:05d}.npz'
        assert Path(entry['path']).resolve() == path and sha(path) == entry['sha256']
        with np.load(path, allow_pickle=False) as a:
            n = entry['rows']
            assert n > 0 and a['beam_tokens'].shape == (n, 20, 4)
            assert a['record_sha256'].shape == (n,) and a['record_sha256'].dtype == np.dtype('S64')
            assert all(len(x) == 64 and all(c in b'0123456789abcdef' for c in x) for x in a['record_sha256'])
            current = set(a['user_ids'].tolist())
            assert not users & current and len(current) == n
            users.update(current)
            np.testing.assert_array_equal(index.lookup_tokens(a['beam_tokens']), a['raw_beam_rows'])
            np.testing.assert_array_equal(unique_beam_rows(a['raw_beam_rows']), a['valid_beam_rows'])
            add_metrics(totals, audit_predictions(a, seen, len(index.keys), n), n)
            identity.update(a['record_sha256'].tobytes())
            count += n
            valid_beams += int((a['valid_beam_rows'] >= 0).sum())
    assert count == len(users) == expected_records and identity.hexdigest() == r['record_identity_sha256']
    assert valid_beams == r['valid_generated_slots']
    for mode, values in totals.items():
        for key, total in values.items(): assert abs(total / count - r['metrics'][mode][key]) < 1e-12
    receipt = dict(status='passed', records=count, report=dict(path=str(report_path), sha256=sha(report_path)),
                   checkpoint=expected_checkpoint, metrics=r['metrics'], test_records_read=False)
    if write_receipt: write_new(directory / 'acceptance.json', receipt)
    return receipt
