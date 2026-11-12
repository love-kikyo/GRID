"""CPU orchestration checks; numerical shard scoring is tested separately."""
from contextlib import nullcontext
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import probe_liger_capacity_validation as probe
from scripts.probe_liger_cpu4 import sha
from src.baselines.liger_state import tree_digest


@pytest.fixture
def harness(tmp_path, monkeypatch):
    from scripts import liger_capacity_runtime
    from src.baselines import liger_data, liger_inference, liger_retrieval
    from src.baselines import liger_sharded_retrieval, liger_efficiency_timing

    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    model = torch.nn.Module()
    model.emb_proj = torch.nn.Linear(2, 2)
    model.predicted_embedding = None
    run_binding = dict(purpose='capacity384_resource_diagnostic_only', microbatch=512)
    state = dict(model=model.state_dict(), binding=run_binding, progress=dict(updates=13))
    checkpoint = tmp_path / 'diagnostic.pt'
    torch.save(state, checkpoint)
    checkpoint_entry = dict(path=str(checkpoint), sha256=sha(checkpoint),
                            full_state_sha256=tree_digest(state))
    train_report = tmp_path / 'training_report.json'
    train_report.write_text('{}')
    receipt = dict(status='passed_resource_training', manifest_sha256=sha(manifest),
        source_git_commit='source-bound', exact_fresh_process_replay=True,
        formal_training_ready=False, test_records_read=False,
        reports=[dict(path=str(train_report), sha256=sha(train_report))],
        validation_checkpoint=checkpoint_entry)
    receipt_path = tmp_path / 'training_acceptance.json'
    receipt_path.write_text(json.dumps(receipt))
    m = dict(phase='capacity384_resource_only', source_git_commit='source-bound', asset_rows=83)
    c = dict(deadline_seconds=2400, validation=dict(chunk_size=65536,
        benchmark=dict(warmup_repeats=2, measured_repeats=3)))
    calls = SimpleNamespace(context=[], caches=[], generation=0, bindings=0, failure=None)

    class Cold:
        def __len__(self): return 23915712
        def __getitem__(self, value):
            assert isinstance(value, slice)
            return np.arange(value.stop, dtype=np.int32)

    def batches(**kwargs):
        assert kwargs == dict(training=False, validation_limit=65)
        yield 0, list(range(32))
        yield 1, list(range(32, 64))
        yield 2, [64]

    ctx = SimpleNamespace(model=model, binding=run_binding, sids=None,
        content=np.ones((83, 2), np.float32), cold=Cold(), seen=np.array([80], dtype=np.int32),
        batches=batches, method={}, official=None,
        index=SimpleNamespace(lookup_tokens=lambda tokens: tokens[..., 0].astype(np.int32)),
        gpus=[dict(name='GPU0'), dict(name='GPU1')], resources=dict(allocated_cpus=4),
        parameter_report=dict(total_params=41868160), cpu_checks=[])
    ctx.cpu_check = lambda: ctx.cpu_checks.append(dict(allowed_cpus=[0, 1, 2, 3]))

    def context(*args, **kwargs):
        calls.context.append(kwargs)
        return ctx

    def binding(path):
        assert path == manifest
        calls.bindings += 1
        return m, c

    def pack(records, *_):
        size = len(records)
        return dict(input_sids=np.asarray(records)[:, None],
                    labels_sids=np.ones((size, 4), np.int64)), dict(
            user_ids=np.asarray(records, dtype=np.int64),
            target_rows=np.ones(size, dtype=np.int32),
            serialized_sha256=[f'{record:064x}' for record in records])

    def generate(actual_model, inputs, *_):
        assert actual_model is model and not model.training
        calls.generation += 1
        n = len(inputs['labels_sids'])
        query = torch.zeros(n, 2)
        if calls.failure == 'gt_leak':
            query[:, 0] = torch.as_tensor(inputs['labels_sids'][:, 0])
        if calls.failure == 'model_mutation':
            with torch.no_grad(): model.emb_proj.weight.add_(1)
        rows = np.tile(np.arange(20, dtype=np.int32), (n, 1))
        return query, dict(beam_tokens=np.repeat(rows[:, :, None], 4, axis=2),
                          raw_beam_rows=rows.copy(), valid_beam_rows=rows.copy())

    def ranked(query, beams):
        scores = np.zeros(beams.shape, np.float32)
        return dict(hybrid_rows=beams.copy(), hybrid_scores=scores.copy(),
                    beam_only_rows=beams.copy(), beam_only_scores=scores.copy(),
                    valid_beam_rows=beams.copy(), beam_scores=scores.copy())

    class Cache:
        def __init__(self, projection, content, cold, **kwargs):
            assert projection is model.emb_proj
            self.full = isinstance(cold, Cold)
            count = len(cold)
            self.shard_sizes = ((count + 1) // 2, count // 2)
            self.cache_bytes = tuple(size * (384 * 4 + 8) for size in self.shard_sizes)
            self.last_scan_rows = 0
            self.closed = False
            calls.caches.append(self)
        def rank(self, query, beams):
            self.last_scan_rows = sum(self.shard_sizes)
            if self.full and calls.failure == 'incomplete_scan': self.last_scan_rows -= 1
            return ranked(query, beams)
        def close(self): self.closed = True
        def __enter__(self): return self
        def __exit__(self, *_): self.close()

    monkeypatch.setattr(probe, 'binding', binding)
    monkeypatch.setattr(liger_capacity_runtime, 'build_context', context)
    monkeypatch.setattr(liger_data, 'pack_batch', pack)
    monkeypatch.setattr(liger_inference, 'generate_queries', generate)
    monkeypatch.setattr(liger_sharded_retrieval, 'ShardedColdCatalogCache', Cache)
    monkeypatch.setattr(liger_retrieval, 'ColdCatalogCache', Cache)
    monkeypatch.setattr(liger_retrieval, 'full_cold_rank', lambda cache, query, beams: ranked(query, beams))
    monkeypatch.setattr(liger_efficiency_timing, 'process_memory_snapshot', lambda: dict(rss_bytes=1))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *a: None)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda *a: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda i: i + 100)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda i: i + 200)
    monkeypatch.setattr(torch.cuda, 'device', lambda *a: nullcontext())
    monkeypatch.setenv('SLURM_JOB_ID', 'test-validation-job')
    return SimpleNamespace(path=manifest, ctx=ctx, calls=calls, c=c,
                           receipt=receipt, receipt_path=receipt_path, checkpoint=checkpoint)


def test_two_gpu_probe_runs_complete_65_records_and_audits_persisted_artifacts(harness):
    h = harness
    probe.run(h.path)
    assert h.calls.context == [dict(microbatch=512, devices=2)]
    assert h.calls.bindings == 2
    report = json.loads((h.path.parent / 'validation_report.json').read_text())
    receipt = json.loads((h.path.parent / 'validation_acceptance.json').read_text())
    assert report['status'] == 'passed' and report['records'] == 65
    assert [row['rows'] for row in report['batches']] == [32, 32, 1]
    assert all(row['scanned_rows'] == 23915712 for row in report['batches'])
    assert report['shard_sizes'] == [11957856, 11957856]
    assert report['small_catalog_single_vs_two_gpu_passed']
    assert report['model_weights_unchanged'] and report['gt_independence_verified']
    assert not report['formal_training_ready'] and not report['formal_checkpoint_produced']
    assert not report['test_records_read'] and not report['automatic_successor']
    assert receipt['report']['sha256'] == sha(h.path.parent / 'validation_report.json')
    assert receipt['status'] == 'passed_resource_validation'
    assert receipt['cold_rows'] == 23915712
    assert len(report['benchmark']['seconds']) == 3
    assert report['benchmark']['repeated_same_inputs']
    assert len(report['artifacts']) == 1 and report['artifacts'][0]['rows'] == 65
    assert all(cache.closed for cache in h.calls.caches if cache.full)
    with np.load(report['artifacts'][0]['path'], allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays['user_ids'], np.arange(65))
        assert arrays['beam_tokens'].shape == (65, 20, 4)


@pytest.mark.parametrize('failure', ['incomplete_scan', 'gt_leak', 'model_mutation'])
def test_validation_failure_never_produces_acceptance_and_closes_full_cache(harness, failure):
    h = harness
    h.calls.failure = failure
    with pytest.raises(AssertionError): probe.run(h.path)
    report = json.loads((h.path.parent / 'validation_report.json').read_text())
    assert report['status'] == 'failed' and 'error' in report
    assert not (h.path.parent / 'validation_acceptance.json').exists()
    assert all(cache.closed for cache in h.calls.caches if cache.full)


@pytest.mark.parametrize('field,value', [
    ('status', 'failed'), ('exact_fresh_process_replay', False),
    ('formal_training_ready', True), ('test_records_read', True),
    ('source_git_commit', 'other-code'), ('manifest_sha256', 'different-manifest'),
])
def test_unaccepted_or_wrong_training_receipt_is_rejected_before_model_allocation(harness, field, value):
    h = harness
    h.receipt[field] = value
    h.receipt_path.write_text(json.dumps(h.receipt))
    with pytest.raises(AssertionError): probe.run(h.path)
    assert h.calls.context == []
    assert not (h.path.parent / 'validation').exists()


def test_checkpoint_byte_change_is_rejected_before_model_allocation(harness):
    h = harness
    with h.checkpoint.open('ab') as handle: handle.write(b'changed')
    with pytest.raises(AssertionError): probe.run(h.path)
    assert h.calls.context == []


@pytest.mark.parametrize('change', ['binding', 'updates', 'state_digest'])
def test_wrong_checkpoint_state_is_rejected_before_inference(harness, change):
    h = harness
    state = torch.load(h.checkpoint, map_location='cpu', weights_only=False)
    if change == 'binding': state['binding']['microbatch'] = 1024
    if change == 'updates': state['progress']['updates'] = 12
    entry = h.receipt['validation_checkpoint']
    entry['full_state_sha256'] = tree_digest(state)
    if change == 'state_digest': entry['full_state_sha256'] = 'wrong'
    torch.save(state, h.checkpoint)
    entry['sha256'] = sha(h.checkpoint)
    h.receipt_path.write_text(json.dumps(h.receipt))
    with pytest.raises(AssertionError): probe.run(h.path)
    assert h.calls.generation == 0
    assert not (h.path.parent / 'validation_acceptance.json').exists()


def test_create_only_output_prevents_overwriting_previous_validation(harness):
    (harness.path.parent / 'validation').mkdir()
    with pytest.raises(AssertionError): probe.run(harness.path)
    assert harness.calls.context == []
