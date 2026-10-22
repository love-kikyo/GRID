import copy
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.baselines import liger_prefix_runtime as runtime
from src.baselines.liger_data import Record, pack_batch
from src.baselines.liger_full_catalog import RowCatalog
from src.baselines.liger_standardization import (
    StandardizedContent, fit_standardization, statistics_digest,
)


def fixture(tmp_path):
    raw = np.arange(21).reshape(7, 3).astype(np.int8)
    seen = np.array([0, 1, 2, 4, 6], dtype=np.int32)
    statistics = fit_standardization(raw, seen, content_sha256='a' * 64)
    path = tmp_path / 'standardization.json'
    path.write_text(json.dumps(statistics))
    manifest = dict(standardization=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
        assets=dict(model_embedding_copy=dict(sha256='a' * 64)),
        catalog=dict(rows=len(seen), sha256='different-npy-file-hash'), asset_rows=len(raw),
        train_files=[dict(path='train-only')], validation_files=[dict(path='validation-only')], seed=42)
    config = dict(training=dict(max_optimizer_steps=19260, max_attempts=3, warmup_steps=963,
                                batch_size=1024, shuffle_buffer=8192))
    return raw, seen, statistics, manifest, config


def mock_base(monkeypatch, raw, seen, config):
    def initialize(self, manifest, *, build_model=True):
        self.m, self.config, self.seen = manifest, copy.deepcopy(config), seen
        self.binding, self.tf = {'original_binding': 'unchanged'}, object()
        if build_model:
            self.content = raw
            self.catalog = RowCatalog(raw, seen)
            self.cold = np.array([3, 5], dtype=np.int32)
    monkeypatch.setattr(runtime.BaseRuntime, '__init__', initialize)
    monkeypatch.setattr(runtime, 'TRAIN_SEEN_ROWS', len(seen))


def test_binding_uses_both_stats_hashes_and_canonical_seen_values(tmp_path):
    raw, seen, stats, manifest, _ = fixture(tmp_path)
    loaded, binding = runtime.read_standardization(manifest, seen, content=raw)
    assert loaded == stats
    assert binding == dict(standardization_sha256=manifest['standardization']['sha256'],
        standardization_stats_sha256=stats['stats_sha256'],
        standardization_seen_rows_sha256=stats['seen_rows_sha256'],
        supervision_tag=runtime.SUPERVISION_TAG)
    assert runtime.seen_rows_digest(seen) == hashlib.sha256(seen.astype('<i8').tobytes()).hexdigest()
    assert runtime.seen_rows_digest(seen.astype(np.int64), chunk_size=1) == stats['seen_rows_sha256']
    assert manifest['catalog']['sha256'] != stats['seen_rows_sha256']


@pytest.mark.parametrize('change', ['file_sha', 'asset_sha', 'row_values', 'row_count', 'asset_rows', 'stats_hash'])
def test_standardization_mismatch_is_rejected(tmp_path, change):
    raw, seen, stats, manifest, _ = fixture(tmp_path)
    if change == 'file_sha': manifest['standardization']['sha256'] = 'b' * 64
    elif change == 'asset_sha': manifest['assets']['model_embedding_copy']['sha256'] = 'b' * 64
    elif change == 'row_values': seen = np.array([0, 1, 2, 3, 6])
    elif change == 'row_count': manifest['catalog']['rows'] += 1
    elif change == 'asset_rows': manifest['asset_rows'] += 1
    else:
        stats['stats_sha256'] = 'b' * 64
        path = tmp_path / 'standardization.json'
        path.write_text(json.dumps(stats))
        manifest['standardization']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError): runtime.read_standardization(manifest, seen, content=raw)


def test_even_self_consistent_statistics_cannot_change_seen_row_fingerprint(tmp_path):
    raw, seen, stats, manifest, _ = fixture(tmp_path)
    stats['seen_rows_sha256'] = 'b' * 64
    stats['stats_sha256'] = statistics_digest(stats)
    path = tmp_path / 'standardization.json'
    path.write_text(json.dumps(stats))
    manifest['standardization']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='training-seen'):
        runtime.read_standardization(manifest, seen, content=raw)


def test_runtime_wraps_once_and_shares_content_for_history_training_and_cold(tmp_path, monkeypatch):
    import torch
    import torch.nn.functional as F
    from src.baselines.liger_retrieval import ColdCatalogCache

    raw, seen, stats, manifest, config = fixture(tmp_path)
    mock_base(monkeypatch, raw, seen, config)
    context = runtime.PrefixRuntime(manifest)
    assert context.raw_content is raw and isinstance(context.content, StandardizedContent)
    assert context.catalog.content is context.content and context.catalog.rows is context.seen
    assert context.binding['original_binding'] == 'unchanged'
    assert context.binding['supervision_tag'] == runtime.SUPERVISION_TAG
    assert context.training_budget == dict(max_optimizer_steps=19260, max_attempts=3)
    expected = ((raw - np.array(stats['mean'])) / np.array(stats['scale'])).astype('float32')
    np.testing.assert_array_equal(context.catalog[:], expected[seen])
    sids = np.arange(28).reshape(7, 4)
    packed, _ = pack_batch([Record(3, (0, 3), 'c' * 64)], sids, context.content)
    np.testing.assert_array_equal(packed['input_embeddings'][0, 0], expected[0])
    assert not packed['input_embeddings'][0, 1:].any()
    projection = torch.nn.Linear(3, 2).eval()
    cache = ColdCatalogCache(projection, context.content, context.cold, device='cpu', chunk_size=1)
    with torch.no_grad(): expected_vectors = F.normalize(projection(torch.from_numpy(expected[context.cold])), dim=-1)
    torch.testing.assert_close(cache.vectors, expected_vectors)


def test_no_model_mode_still_validates_binding_without_tf_or_model(tmp_path, monkeypatch):
    raw, seen, _, manifest, config = fixture(tmp_path)
    mock_base(monkeypatch, raw, seen, config)
    context = runtime.PrefixRuntime(manifest, build_model=False)
    assert not hasattr(context, 'content') and 'standardization_sha256' in context.binding


def test_validation_routes_to_old_unexpanded_stream_but_training_to_prefix_stream(tmp_path, monkeypatch):
    raw, seen, _, manifest, config = fixture(tmp_path)
    mock_base(monkeypatch, raw, seen, config)
    context = runtime.PrefixRuntime(manifest)
    calls = []
    def old_batches(self, **kwargs):
        calls.append(('validation', kwargs, self.content))
        return [(0, ['unexpanded original records'])]
    monkeypatch.setattr(runtime.BaseRuntime, 'batches', old_batches)
    assert context.batches(training=False, validation_limit=64) == [(0, ['unexpanded original records'])]
    assert calls[0][1] == dict(training=False, epoch=0, start_batch=0, validation_limit=64)
    assert calls[0][2] is context.content
    parents = [Record(1, (0, 1, 2), 'a' * 64)]
    closed = []
    def source(tf, files, asset_rows):
        assert files == manifest['train_files'] and asset_rows == len(raw)
        try: yield from parents
        finally: closed.append(True)
    def prefix_batches(entries, stream_factory, **kwargs):
        calls.append(('training', kwargs, list(stream_factory(entries))))
        return ['prefix stream result']
    monkeypatch.setattr(runtime, 'record_stream', source)
    monkeypatch.setitem(sys.modules, 'src.baselines.liger_prefix_stream', SimpleNamespace(prefix_epoch_batches=prefix_batches))
    assert context.batches(training=True, epoch=4, start_batch=8) == ['prefix stream result']
    assert calls[1][1] == dict(batch_size=1024, seed=42, epoch=4, shuffle_buffer=8192, start_batch=8,
                              history_items=7, prefix_shuffle_buffer=8192)
    assert calls[1][2] == parents and closed == [True]
    context.config['training']['prefix_shuffle_buffer'] = 17
    context.batches(training=True)
    assert calls[-1][1]['prefix_shuffle_buffer'] == 17
    with pytest.raises(ValueError, match='Cannot cap'): context.batches(training=True, validation_limit=1)


@pytest.mark.parametrize('key,value', [('max_attempts', None), ('max_attempts', True),
    ('max_optimizer_steps', 0), ('warmup_steps', 19260), ('batch_size', 0),
    ('shuffle_buffer', 0), ('prefix_shuffle_buffer', 0), ('history_items', 8)])
def test_fixed_budget_and_supervision_constraints_are_explicit(tmp_path, key, value):
    _, _, _, _, config = fixture(tmp_path)
    config['training'][key] = value
    with pytest.raises(ValueError): runtime.training_budget(config)


def test_formal_runtime_rejects_diagnostic_catalog_size(tmp_path, monkeypatch):
    raw, seen, _, manifest, config = fixture(tmp_path)
    mock_base(monkeypatch, raw, seen, config)
    monkeypatch.setattr(runtime, 'TRAIN_SEEN_ROWS', 11542755)
    with pytest.raises(ValueError, match='frozen train-seen'): runtime.PrefixRuntime(manifest)


@pytest.mark.parametrize('rows', [np.array([], dtype='int64'), np.array([1, 0]),
    np.array([0, 0]), np.array([-1, 1]), np.array([0., 1.]), np.array([True]),
    np.array([0, 2 ** 63], dtype=np.uint64)])
def test_bad_canonical_seen_rows_are_rejected(rows):
    with pytest.raises(ValueError): runtime.seen_rows_digest(rows)
