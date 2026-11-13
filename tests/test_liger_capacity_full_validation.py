"""CPU orchestration and independent-artifact tests for the two-GPU evaluator."""
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import validate_liger_capacity_cpu4 as val
from scripts.probe_liger_cpu4 import sha
from src.baselines.liger_prefix_training import initial_progress
from src.baselines.liger_state import tree_digest


@pytest.fixture
def harness(tmp_path, monkeypatch):
    from src.baselines import liger_efficiency_timing
    model = torch.nn.Module()
    model.emb_proj = torch.nn.Linear(2, 2)
    model.predicted_embedding = None
    calls = SimpleNamespace(caches=[], generation=0, failure=None, closed=False, limit=None)

    class Cold:
        def __len__(self): return 23915712

    def batches(**kwargs):
        assert kwargs['training'] is False
        calls.limit = kwargs['validation_limit']
        try:
            yield 0, list(range(32))
            yield 1 if calls.failure != 'bad_cursor' else 2, list(range(32, 64))
            if calls.failure != 'missing_tail':
                yield 2, [64] if calls.failure != 'extra_rows' else [64, 65]
        finally:
            calls.closed = True

    ctx = SimpleNamespace(model=model, binding=dict(purpose='capacity384_training_gate', microbatch=1024),
        sids=None, content=np.ones((83, 2), np.float32), cold=Cold(), seen=np.array([80], dtype=np.int32),
        batches=batches, method={}, official=None,
        index=SimpleNamespace(keys=np.arange(83), lookup_tokens=lambda tokens: tokens[..., 0].astype(np.int32)),
        gpus=[dict(name='NVIDIA RTX 4090', total_bytes=24*2**30)]*2,
        resources=dict(allocated_cpus=4, library_threads=1, allowed_cpus=[0, 1, 2, 3]), cpu_checks=[],
        scheduler=SimpleNamespace(state_dict=lambda: dict(total=22460, warmup=1123, base_lr=.0003, updates=0)),
        optimizer=object(), scaler=object())
    ctx.cpu_check = lambda: ctx.cpu_checks.append(dict(allowed_cpus=[0, 1, 2, 3], processes=[dict(threads_checked=4)]))

    def pack(records, *_):
        users = np.asarray(records, dtype=np.int64)
        if calls.failure == 'duplicate_user' and records[0] == 32: users[0] = 0
        if calls.failure == 'duplicate_in_batch': users[1] = users[0]
        return dict(input_sids=np.asarray(records)[:, None], labels_sids=np.ones((len(records), 4), np.int64)), dict(
            user_ids=users, target_rows=np.ones(len(records), dtype=np.int32),
            serialized_sha256=[f'{record:064x}' for record in records])

    def generate(actual, inputs, *_):
        assert actual is model and not model.training
        calls.generation += 1
        size = len(inputs['labels_sids'])
        query = torch.zeros(size, 2)
        if calls.failure == 'gt_leak': query[:, 0] = torch.as_tensor(inputs['labels_sids'][:, 0])
        if calls.failure == 'model_mutation':
            with torch.no_grad(): model.emb_proj.weight.add_(1)
        rows = np.tile(np.arange(20, dtype=np.int32), (size, 1))
        return query, dict(beam_tokens=np.repeat(rows[:, :, None], 4, axis=2),
            raw_beam_rows=rows.copy(), valid_beam_rows=rows.copy())

    class Cache:
        def __init__(self, projection, content, cold, **kwargs):
            assert projection is model.emb_proj and not projection.training and kwargs == dict(chunk_size=65536)
            self.shard_sizes = (11957856, 11957856)
            self.cache_bytes = (18462929664, 18462929664)
            self.last_scan_rows, self.closed = 0, False
            calls.caches.append(self)
        def rank(self, query, beams):
            self.last_scan_rows = 23915712 - (calls.failure == 'incomplete_scan')
            if calls.failure == 'rank_error': raise RuntimeError('fake worker failure')
            scores = np.zeros(beams.shape, np.float32)
            return dict(hybrid_rows=beams.copy(), hybrid_scores=scores.copy(),
                beam_only_rows=beams.copy(), beam_only_scores=scores.copy(),
                valid_beam_rows=beams.copy(), beam_scores=scores.copy())
        def close(self): self.closed = True

    monkeypatch.setattr(val, 'pack_batch', pack)
    monkeypatch.setattr(val, 'generate_queries', generate)
    monkeypatch.setattr(val, 'ShardedColdCatalogCache', Cache)
    monkeypatch.setattr(liger_efficiency_timing, 'process_memory_snapshot', lambda: dict(rss_bytes=1))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *args: None)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda *args: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda i: i+100)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda i: i+200)
    monkeypatch.setattr(torch.cuda, 'device', lambda *args: nullcontext())
    monkeypatch.setenv('SLURM_JOB_ID', 'validation-job')
    return SimpleNamespace(ctx=ctx, calls=calls, directory=tmp_path/'validation', checkpoint=dict(path='unused', sha256='bound'))


def test_streaming_65_tails_full_cold_audit_timing_and_coverage(harness):
    h = harness
    before = tree_digest(h.ctx.model.state_dict())
    report, accepted, proof = val.evaluate_context(h.ctx, h.directory, 65, h.checkpoint, validation_limit=65)
    assert h.calls.limit == 65 and h.calls.closed and all(cache.closed for cache in h.calls.caches)
    assert report['records'] == 65 and report['batches'] == 3 and report['cold_rows'] == 23915712
    assert report['raw_generation_coverage']['hits'] == {'5': 65, '10': 65, '20': 65}
    assert report['raw_generation_coverage']['recall'] == {'5': 1., '10': 1., '20': 1.}
    assert report['metrics']['beam_only']['Recall@5'] == 1.
    assert accepted['status'] == 'passed' and accepted['report']['sha256'] == sha(h.directory/'report.json')
    assert proof['model_weights_unchanged'] and proof['before_model_sha256'] == proof['after_model_sha256'] == before
    assert proof['load_scope'] == 'model_weights_only' and proof['audit_seconds'] > 0
    assert h.ctx.model.training and h.ctx.model.predicted_embedding is None
    assert len(report['artifacts']) == 1 and report['artifacts'][0]['rows'] == 65
    with np.load(report['artifacts'][0]['path'], allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays['user_ids'], np.arange(65))
    steps = [json.loads(line) for line in (h.directory/'batches.jsonl').read_text().splitlines()]
    assert [step['rows'] for step in steps] == [32, 32, 1]
    assert all(step['scanned_rows'] == 23915712 for step in steps)
    assert steps[0]['gt_independence_seconds'] > 0 and steps[1]['gt_independence_seconds'] == 0


@pytest.mark.parametrize('failure', ['incomplete_scan', 'gt_leak', 'model_mutation', 'duplicate_user',
    'duplicate_in_batch', 'bad_cursor', 'missing_tail', 'extra_rows', 'rank_error'])
def test_failed_validation_has_no_acceptance_and_closes_live_resources(harness, failure):
    h = harness
    h.calls.failure = failure
    with pytest.raises((AssertionError, RuntimeError)):
        val.evaluate_context(h.ctx, h.directory, 65, h.checkpoint, validation_limit=65)
    assert not (h.directory/'acceptance.json').exists()
    assert json.loads((h.directory/'failure.json').read_text())['status'] == 'failed'
    assert h.calls.closed and all(cache.closed for cache in h.calls.caches)
    assert h.ctx.model.training


def test_validation_outputs_are_create_only(harness):
    h = harness
    h.directory.mkdir()
    with pytest.raises(FileExistsError): val.evaluate_context(h.ctx, h.directory, 65, h.checkpoint)
    assert h.calls.generation == 0


def test_cpu_audit_keeps_matching_int32_population_and_persisted_ids(harness, monkeypatch):
    h = harness
    seen_before = h.ctx.seen.copy()
    original, calls = val.audit, []
    def audit(directory, seen, *args, **kwargs):
        assert seen is h.ctx.seen and seen.dtype == np.dtype('int32')
        np.testing.assert_array_equal(seen, seen_before)
        calls.append(seen)
        with np.load(h.directory/'part-00000.npz', allow_pickle=False) as arrays:
            assert arrays['hybrid_rows'].dtype == arrays['beam_only_rows'].dtype == seen.dtype
        return original(directory, seen, *args, **kwargs)
    monkeypatch.setattr(val, 'audit', audit)
    val.evaluate_context(h.ctx, h.directory, 65, h.checkpoint, validation_limit=65)
    assert len(calls) == 1 and h.ctx.seen.dtype == np.dtype('int32')
    np.testing.assert_array_equal(h.ctx.seen, seen_before)


@pytest.mark.parametrize('count,tail,shards', [(88, 24, [88]), (3233, 1, [3200, 33])])
def test_real_stream_tail24_and_prediction_shard_boundary(harness, count, tail, shards):
    h = harness
    def batches(**kwargs):
        assert kwargs == dict(training=False, validation_limit=None)
        for batch, start in enumerate(range(0, count, 32)):
            yield batch, list(range(start, min(start+32, count)))
    h.ctx.batches = batches
    report, accepted, _ = val.evaluate_context(h.ctx, h.directory, count, h.checkpoint)
    assert report['records'] == accepted['records'] == count
    steps = [json.loads(line) for line in (h.directory/'batches.jsonl').read_text().splitlines()]
    assert steps[-1]['rows'] == tail
    assert [entry['rows'] for entry in report['artifacts']] == shards


@pytest.mark.parametrize('tamper', ['raw_coverage', 'memory', 'cpu', 'shard', 'timing', 'log_bytes'])
def test_independent_extra_audit_rejects_tampered_claims(harness, tamper):
    h = harness
    val.evaluate_context(h.ctx, h.directory, 65, h.checkpoint, validation_limit=65)
    path = h.directory/'report.json'
    r = json.loads(path.read_text())
    if tamper == 'raw_coverage': r['raw_generation_coverage']['hits']['20'] -= 1
    elif tamper == 'memory': r['peak_reserved_bytes'][1] = 100*2**30
    elif tamper == 'cpu': r['resources']['allocated_cpus'] = 8
    elif tamper == 'shard': r['shard_sizes'][0] -= 1
    elif tamper == 'timing': r['timing']['packing_seconds'] += 1
    else:
        with (h.directory/'batches.jsonl').open('a') as handle: handle.write('{}\n')
    path.write_text(json.dumps(r))
    with pytest.raises(AssertionError): val.audit_extended(h.directory, h.ctx, 65, h.checkpoint)


def _checkpoint(h, tmp_path, *, full=False):
    p = initial_progress(max_optimizer_steps=22460, max_attempts=22460)
    attempts, prefixes, finals = (22460, 45996113, 6573674) if full else (3, 6144, 850)
    p.update(next_batch=attempts, samples_in_epoch=prefixes, batches_consumed=attempts,
             attempts=attempts, prefix_count=prefixes, final_count=finals, updates=attempts)
    state = dict(format_version=1, binding=copy.deepcopy(h.ctx.binding), model=h.ctx.model.state_dict(),
        progress=p, model_training=True, scheduler=dict(total=22460, warmup=1123, base_lr=.0003, updates=attempts),
        optimizer=dict(sentinel='not restored'), scaler=dict(sentinel='not restored'), rng=dict(cuda=['not restored']))
    path = tmp_path/'checkpoint.pt'
    torch.save(state, path)
    entry = dict(path=str(path), sha256=sha(path), full_state_sha256=tree_digest(state),
                 **{key:p[key] for key in ('epoch', 'next_batch', 'updates', 'attempts')})
    return state, entry


def test_checkpoint_bridge_restores_model_only_not_optimizer_or_rng(harness, tmp_path):
    h = harness
    state, entry = _checkpoint(h, tmp_path)
    expected = tree_digest(state['model'])
    optimizer, scaler, scheduler = h.ctx.optimizer, h.ctx.scaler, h.ctx.scheduler
    rng = torch.get_rng_state().clone()
    with torch.no_grad(): h.ctx.model.emb_proj.weight.add_(1)
    loaded = val.load_model_checkpoint(h.ctx, entry, expected_progress=state['progress'])
    assert loaded['model_state_sha256'] == expected and not h.ctx.model.training
    assert loaded['load_scope'] == 'model_weights_only'
    assert h.ctx.optimizer is optimizer and h.ctx.scaler is scaler and h.ctx.scheduler is scheduler
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)


@pytest.mark.parametrize('field', ['bytes', 'state_digest', 'binding', 'model_training', 'scheduler',
                                  'entry_cursor', 'nonfinite', 'expected_progress'])
def test_checkpoint_bridge_rejects_wrong_identity_or_state(harness, tmp_path, field):
    h = harness
    state, entry = _checkpoint(h, tmp_path)
    expected = copy.deepcopy(state['progress'])
    if field == 'binding': state['binding']['purpose'] = 'wrong-purpose'
    elif field == 'model_training': state['model_training'] = False
    elif field == 'scheduler': state['scheduler']['updates'] -= 1
    elif field == 'nonfinite': next(iter(state['model'].values())).fill_(float('nan'))
    elif field == 'expected_progress': expected['updates'] -= 1
    torch.save(state, entry['path'])
    entry.update(sha256=sha(entry['path']), full_state_sha256=tree_digest(state))
    if field == 'bytes':
        with open(entry['path'], 'ab') as handle: handle.write(b'changed')
    elif field == 'state_digest': entry['full_state_sha256'] = 'wrong'
    elif field == 'entry_cursor': entry['updates'] -= 1
    with pytest.raises(AssertionError): val.load_model_checkpoint(h.ctx, entry, expected_progress=expected)


@pytest.fixture
def formal_harness(harness, tmp_path, monkeypatch):
    import scripts
    h = harness
    directory = tmp_path/'formal'
    directory.mkdir()
    state, entry = _checkpoint(h, tmp_path, full=True)
    (directory/'checkpoints').mkdir()
    final_path = directory/'checkpoints/pass01-before-validation.pt'
    final_path.write_bytes(Path(entry['path']).read_bytes())
    entry['path'] = str(final_path)
    path = tmp_path/'manifest.json'
    path.write_text('{}')
    m = dict(phase='capacity384_first_pass_formal', formal_training_ready=True,
        source_git_commit='bound-source', expected_initial_state_sha256='fresh-initial', run_directory=str(directory))
    c = dict(training=dict(batches_per_epoch=22460, prefixes_per_epoch=45996113, parents_per_epoch=6573674),
             validation=dict(records=347416))
    for name in ('training_result.json', 'steps.jsonl'): (directory/name).write_text('{}')
    counts = {key:state['progress'][key] for key in ('attempts', 'updates', 'skipped_updates',
                                                   'prefix_count', 'final_count', 'data_chain')}
    receipt = dict(status='passed_training', awaiting_validation=True, job_id='train-job',
        source_git_commit=m['source_git_commit'], manifest_sha256=sha(path),
        test_records_read=False, automatic_successor=False,
        initial_state_sha256=m['expected_initial_state_sha256'], checkpoint_binding=h.ctx.binding,
        report=dict(path=str(directory/'training_result.json'), sha256=sha(directory/'training_result.json')),
        steps=dict(path=str(directory/'steps.jsonl'), sha256=sha(directory/'steps.jsonl')),
        counts=counts, checkpoint=entry)
    receipt_path = directory/'training_acceptance.json'
    receipt_path.write_text(json.dumps(receipt))
    calls = SimpleNamespace(context=[], evaluate=[])

    def context(manifest, **kwargs):
        assert manifest == m
        calls.context.append(kwargs)
        return h.ctx

    def evaluate(ctx, output, count, checkpoint, **kwargs):
        assert ctx is h.ctx and checkpoint == entry and count == 347416
        calls.evaluate.append(dict(count=count, **kwargs))
        output.mkdir()
        report = dict(metrics={'hybrid': {'Recall@5': .1, 'NDCG@5': .08}},
                      raw_generation_coverage=dict(hits={'20': 10}, recall={'20': 10/count}))
        (output/'report.json').write_text(json.dumps(report))
        accepted = dict(status='passed', report=dict(path=str(output/'report.json'), sha256=sha(output/'report.json')))
        (output/'acceptance.json').write_text(json.dumps(accepted))
        return report, accepted, dict(model_weights_unchanged=True)

    contract = SimpleNamespace(FORMAL_PHASE=m['phase'], binding=lambda actual: (m, c))
    trainer = SimpleNamespace(build_context=context)
    monkeypatch.setitem(sys.modules, 'scripts.liger_capacity_training_manifest', contract)
    monkeypatch.setitem(sys.modules, 'scripts.train_liger_capacity_cpu4', trainer)
    monkeypatch.setattr(scripts, 'liger_capacity_training_manifest', contract, raising=False)
    monkeypatch.setattr(scripts, 'train_liger_capacity_cpu4', trainer, raising=False)
    monkeypatch.setattr(val, 'evaluate_context', evaluate)
    return SimpleNamespace(path=path, directory=directory, m=m, c=c, receipt=receipt,
                           receipt_path=receipt_path, calls=calls, entry=entry)


def test_formal_run_accepts_single_pass_model_and_stops(formal_harness):
    h = formal_harness
    val.run(h.path)
    assert h.calls.context == [dict(devices=2)] and h.calls.evaluate == [dict(count=347416)]
    result = json.loads((h.directory/'result.json').read_text())
    accepted = json.loads((h.directory/'acceptance.json').read_text())
    assert result['status'] == accepted['status'] == 'passed'
    assert accepted['full_validation_records'] == 347416 and accepted['completed_passes'] == 1
    assert result['selection_scope'].startswith('single authorized pass')
    assert accepted['requires_author_acceptance_before_next_pass']
    assert not accepted['test_records_read'] and not accepted['automatic_successor']
    assert sha(h.entry['path']) == h.entry['sha256']
    assert result['training_progress']['epoch'] == 0


@pytest.mark.parametrize('field,value', [('status', 'failed'), ('awaiting_validation', False),
    ('test_records_read', True), ('automatic_successor', True), ('source_git_commit', 'other'),
    ('manifest_sha256', 'other'), ('initial_state_sha256', 'old-diagnostic')])
def test_formal_run_rejects_unaccepted_training_before_gpu(formal_harness, field, value):
    h = formal_harness
    h.receipt[field] = value
    h.receipt_path.write_text(json.dumps(h.receipt))
    with pytest.raises(AssertionError): val.run(h.path)
    assert not h.calls.context and not h.calls.evaluate


@pytest.mark.parametrize('field', ['prefix_count', 'final_count', 'attempts'])
def test_formal_run_rejects_incomplete_pass_before_gpu(formal_harness, field):
    h = formal_harness
    h.receipt['counts'][field] -= 1
    h.receipt_path.write_text(json.dumps(h.receipt))
    with pytest.raises(AssertionError): val.run(h.path)
    assert not h.calls.context


def test_formal_run_never_overwrites_a_previous_result(formal_harness):
    h = formal_harness
    (h.directory/'result.json').write_text('previous')
    with pytest.raises(AssertionError): val.run(h.path)
    assert (h.directory/'result.json').read_text() == 'previous' and not h.calls.context
