import copy
import hashlib
import inspect
import json

import numpy as np
import pytest

from src.baselines.liger_feature_cache import build_cached_seen, verify_cached_seen
from src.baselines.liger_inputs import pack_sequences
from src.baselines.liger_standardization import StandardizedContent, fit_standardization


class TracedContent(StandardizedContent):
    def __init__(self, raw, stats):
        super().__init__(raw, stats, expected_content_sha256='a' * 64)
        self.reads = []

    def __getitem__(self, selection):
        self.reads.append(np.asarray(selection).copy())
        return super().__getitem__(selection)


def fixture():
    raw = np.random.default_rng(31).integers(-120, 121, (13, 5), dtype=np.int16).astype(np.int8)
    raw[:, -1] = 17
    seen = np.array([0, 2, 4, 7, 10], dtype=np.int32)
    stats = fit_standardization(raw, seen, chunk_size=3, content_sha256='a' * 64)
    content = TracedContent(raw, stats)
    binding = dict(standardization_stats_sha256=stats['stats_sha256'],
                   seen_rows_sha256=stats['seen_rows_sha256'], source_sha256='a' * 64)
    return raw, seen, content, binding


def built(chunk_size=2):
    raw, seen, content, binding = fixture()
    wrapper, catalog, summary = build_cached_seen(content, seen, chunk_size=chunk_size, expected_binding=binding)
    return raw, seen, content, binding, wrapper, catalog, summary


@pytest.mark.parametrize('chunk_size', [1, 2, 4, 20, np.int64(3)])
def test_build_only_seen_rows_with_one_fp32_matrix_and_independent_bitwise_verification(chunk_size):
    raw, seen, content, binding, wrapper, catalog, summary = built(chunk_size)
    read_rows = np.concatenate(content.reads)
    np.testing.assert_array_equal(read_rows, seen)
    assert max(map(len, content.reads)) <= chunk_size
    assert catalog.content is wrapper and catalog.rows is seen and wrapper.seen_rows is seen
    assert wrapper.base_content is content and wrapper.shape == raw.shape and wrapper.ndim == 2
    assert len(wrapper) == len(raw) and len(catalog) == len(seen)
    assert wrapper.dtype == np.float32 and wrapper.cache.dtype == np.float32
    assert summary['cache_bytes'] == wrapper.cache.nbytes == len(seen) * raw.shape[1] * 4
    assert summary['cache_bytes'] < raw.size * 4
    expected = content[seen]
    np.testing.assert_array_equal(wrapper.cache.view(np.uint32), expected.view(np.uint32))
    assert summary['cache_sha256'] == hashlib.sha256(expected.astype('<f4').tobytes()).hexdigest()
    assert not wrapper.cache.flags.writeable and not seen.flags.writeable
    content.reads.clear()
    receipt = verify_cached_seen(wrapper, catalog, expected_binding=binding, chunk_size=3)
    json.dumps(summary)
    json.dumps(receipt)
    np.testing.assert_array_equal(np.concatenate(content.reads), seen)
    assert max(map(len, content.reads)) <= 3
    assert receipt['status'] == 'passed' and receipt['feature_values_bitwise_equal']
    assert receipt['cache_sha256'] == summary['cache_sha256']
    for key, value in binding.items(): assert summary[key] == receipt[key] == value


@pytest.mark.parametrize('selection', [0, np.int64(10), [0, 2, 0, 4], [1, 3, 12],
    [0, 1, 10, 12, 2, 1], np.array([[0, 12], [7, 3]]), slice(None), slice(2, 11, 2),
    slice(0, 0), np.array([], dtype=np.int64), np.empty((2, 0), dtype=np.int32)])
def test_global_index_results_match_original_exactly(selection):
    _, _, content, _, wrapper, _, _ = built()
    expected = StandardizedContent.__getitem__(content, selection)
    actual = wrapper[selection]
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_seen_requests_never_recompute_and_mixed_requests_recompute_only_cold_rows():
    _, _, content, _, wrapper, _, _ = built()
    content.reads.clear()
    wrapper[[0, 2, 4, 10, 0]]
    assert content.reads == []
    wrapper[[0, 1, 10, 12, 2, 1]]
    assert len(content.reads) == 1
    np.testing.assert_array_equal(content.reads[0], [1, 12, 1])


def test_catalog_direct_slices_bypass_binary_search_and_return_readonly_shared_storage(monkeypatch):
    _, seen, content, _, wrapper, catalog, _ = built()
    content.reads.clear()
    def forbidden(*args, **kwargs): raise AssertionError('Full seen chunks must not binary-search global IDs')
    monkeypatch.setattr(np, 'searchsorted', forbidden)
    selected = catalog[1:4]
    assert np.shares_memory(selected, wrapper.cache) and not selected.flags.writeable
    np.testing.assert_array_equal(selected, wrapper.cache[1:4])
    assert not content.reads and catalog.rows is seen
    with pytest.raises(ValueError): selected[0, 0] = 0
    with pytest.raises(ValueError): wrapper.cache.flags.writeable = True
    with pytest.raises(ValueError): seen[0] = 1


@pytest.mark.parametrize('selection', [-1, slice(None, None, -1), [2, 0, 2],
    np.array([True, False, True, False, True]), None, (slice(None),),
    (Ellipsis, None), (None, Ellipsis, None), (), (1, None), (True,),
    (np.array([2, 0]), None)])
def test_catalog_position_indexing_matches_existing_rowcatalog(selection):
    from src.baselines.liger_full_catalog import RowCatalog
    _, seen, content, _, _, catalog, _ = built()
    expected = RowCatalog(content, seen)[selection]
    np.testing.assert_array_equal(catalog[selection].view(np.uint32), expected.view(np.uint32))


def test_catalog_does_not_allow_accidental_column_selection():
    *_, catalog, _ = built()
    with pytest.raises(IndexError): catalog[:, 0]


@pytest.mark.parametrize('selection', [-1, 13, [13], [-1, 0], [1.], True, [False],
    (slice(None), 0), slice(-1, None), slice(None, 14), slice(None, None, -1), slice(None, None, 0)])
def test_invalid_global_indexing_is_rejected(selection):
    *_, wrapper, _, _ = built()
    with pytest.raises(ValueError): wrapper[selection]


@pytest.mark.parametrize('rows', [np.array([], dtype=np.int32), np.array([0, 1], dtype=np.int64),
    np.array([1, 0], dtype=np.int32), np.array([0, 0], dtype=np.int32),
    np.array([-1, 0], dtype=np.int32), np.array([0, 13], dtype=np.int32),
    np.array([0., 1.]), np.array([[0, 1]], dtype=np.int32)])
def test_bad_seen_row_layout_is_rejected_before_cache_allocation(rows):
    _, _, content, binding = fixture()
    with pytest.raises(ValueError): build_cached_seen(content, rows, expected_binding=binding)
    assert not content.reads


@pytest.mark.parametrize('key', ['standardization_stats_sha256', 'seen_rows_sha256', 'source_sha256'])
def test_wrong_binding_is_rejected_before_build_and_at_verify(key):
    _, seen, content, binding = fixture()
    wrong = dict(binding, **{key: 'b' * 64})
    with pytest.raises(ValueError, match='binding'):
        build_cached_seen(content, seen, expected_binding=wrong)
    assert not content.reads
    wrapper, catalog, _ = build_cached_seen(content, seen, expected_binding=binding)
    with pytest.raises(ValueError): verify_cached_seen(wrapper, catalog, expected_binding=wrong)


def test_different_seen_set_cannot_claim_unrelated_but_individually_valid_stats_hash():
    _, _, content, binding = fixture()
    other_seen = np.array([0, 2, 4, 7, 11], dtype=np.int32)
    forged = dict(binding, seen_rows_sha256=hashlib.sha256(other_seen.astype('<i8').tobytes()).hexdigest())
    with pytest.raises(ValueError, match='reproduce the bound statistics'):
        build_cached_seen(content, other_seen, expected_binding=forged)


@pytest.mark.parametrize('bad_binding', [None, {}, {'unexpected': 'a' * 64},
    dict(standardization_stats_sha256='A' * 64, seen_rows_sha256='a' * 64, source_sha256='a' * 64)])
def test_binding_schema_and_digest_encoding_are_strict(bad_binding):
    _, seen, content, _ = fixture()
    with pytest.raises(ValueError): build_cached_seen(content, seen, expected_binding=bad_binding)
    assert not content.reads


def test_seen_order_is_checked_across_stream_chunk_boundaries():
    _, _, content, binding = fixture()
    for rows in ([0, 2, 2, 7, 10], [0, 2, 1, 7, 10]):
        with pytest.raises(ValueError, match='sorted unique'):
            build_cached_seen(content, np.array(rows, dtype=np.int32), chunk_size=2, expected_binding=binding)


def test_nonfinite_source_is_rejected_without_publishing_a_cache():
    raw = np.arange(30, dtype=np.float32).reshape(10, 3)
    seen = np.array([0, 3, 8], dtype=np.int32)
    stats = fit_standardization(raw, seen, content_sha256='a' * 64)
    content = StandardizedContent(raw, stats, expected_content_sha256='a' * 64)
    binding = dict(standardization_stats_sha256=stats['stats_sha256'],
                   seen_rows_sha256=stats['seen_rows_sha256'], source_sha256='a' * 64)
    raw[3, 0] = np.nan
    with pytest.raises(ValueError): build_cached_seen(content, seen, expected_binding=binding, chunk_size=2)


def test_tampered_cache_and_mismatched_pair_are_detected():
    _, _, _, binding, wrapper, catalog, _ = built()
    *_, other_wrapper, other_catalog, _ = built()
    with pytest.raises(ValueError, match='identities'):
        verify_cached_seen(wrapper, other_catalog, expected_binding=binding)
    # Deliberate private-owner corruption simulates memory/storage damage;
    # normal clients only receive a read-only public cache view.
    wrapper._values.flags.writeable = True
    wrapper._values[0, 0] += 1
    wrapper._values.flags.writeable = False
    with pytest.raises(ValueError, match='recomputation'):
        verify_cached_seen(wrapper, catalog, expected_binding=binding)
    assert other_wrapper is not wrapper


def test_mutated_original_binding_is_rejected_on_both_read_paths():
    _, _, content, _, wrapper, catalog, _ = built()
    content.stats_sha256 = 'f' * 64
    with pytest.raises(ValueError, match='binding changed'): wrapper[[0, 1]]
    with pytest.raises(ValueError, match='binding changed'): catalog[:]


@pytest.mark.parametrize('chunk', [0, -1, 1.5, True])
def test_invalid_chunk_sizes_are_rejected(chunk):
    _, seen, content, binding = fixture()
    with pytest.raises(ValueError): build_cached_seen(content, seen, chunk_size=chunk, expected_binding=binding)
    wrapper, catalog, _ = build_cached_seen(content, seen, expected_binding=binding)
    with pytest.raises(ValueError): verify_cached_seen(wrapper, catalog, chunk_size=chunk, expected_binding=binding)


def test_raw_or_already_cached_content_cannot_be_reprocessed():
    raw, seen, content, binding, wrapper, _, _ = built()
    for invalid in (raw, wrapper):
        with pytest.raises(ValueError): build_cached_seen(invalid, seen, expected_binding=binding)


def test_padding_real_row_zero_and_masked_catalog_size_are_unchanged():
    raw, seen, content, _, wrapper, catalog, _ = built()
    sid = np.arange(len(raw) * 4).reshape(-1, 4)
    original = pack_sequences([[0, 1, 2], [7, 10]], sid, content)
    cached = pack_sequences([[0, 1, 2], [7, 10]], sid, wrapper)
    for name in ('input_sids', 'attention_mask_sids', 'labels_sids', 'input_embeddings', 'target_rows'):
        np.testing.assert_array_equal(cached[name], original[name])
    assert np.count_nonzero(cached['input_embeddings'][0, 0]) > 0
    assert np.count_nonzero(cached['input_embeddings'][0, 2:]) == 0
    assert np.count_nonzero(cached['input_embeddings'][1, 1:]) == 0
    assert len(wrapper) - len(catalog) == len(raw) - len(seen)


def test_complete_loss_and_every_gradient_are_identical_on_tiny_catalog():
    import torch
    from src.baselines.liger_full_catalog import RowCatalog, full_catalog_cross_entropy

    _, seen, content, _, wrapper, catalog, _ = built()
    torch.manual_seed(83)
    projection = torch.nn.Linear(5, 4)
    reference = copy.deepcopy(projection)
    query = torch.randn(3, 4, requires_grad=True)
    other_query = query.detach().clone().requires_grad_(True)
    labels = torch.tensor([0, 2, 4])
    lazy = full_catalog_cross_entropy(query, labels, projection, RowCatalog(content, seen),
                                     chunk_size=2, masked_count=len(content) - len(seen))
    cached = full_catalog_cross_entropy(other_query, labels, reference, catalog,
                                       chunk_size=2, masked_count=len(wrapper) - len(seen))
    torch.testing.assert_close(lazy, cached, atol=0, rtol=0)
    gradients = torch.autograd.grad(lazy, (query, *projection.parameters()))
    other_gradients = torch.autograd.grad(cached, (other_query, *reference.parameters()))
    for a, b in zip(gradients, other_gradients): torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_module_has_no_learned_projection_or_torch_tensor_cache():
    from src.baselines import liger_feature_cache
    source = inspect.getsource(liger_feature_cache)
    assert 'import torch' not in source and 'import tensorflow' not in source
    assert 'model.emb_proj' not in source
