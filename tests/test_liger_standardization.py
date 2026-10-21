import copy
import hashlib
import json

import numpy as np
import pytest

from src.baselines.liger_standardization import (
    StandardizedContent, fit_standardization, statistics_digest, validate_standardization,
)


def fixture():
    content = np.random.default_rng(42).integers(-127, 128, (257, 9), dtype=np.int16).astype(np.int8)
    content[:, -1] = 13
    seen = np.arange(0, len(content), 3, dtype=np.int32)
    return content, seen


@pytest.mark.parametrize('chunk_size', [1, 7, 64, 1024])
def test_population_moments_match_independent_numpy_oracle_without_optional_dependencies(chunk_size):
    content, seen = fixture()
    selected = content[seen].astype(np.float64)
    mean, var = selected.mean(axis=0), selected.var(axis=0, ddof=0)
    scale = np.sqrt(var)
    scale[scale == 0] = 1.
    stats = fit_standardization(content, seen, chunk_size=chunk_size)
    np.testing.assert_allclose(stats['mean'], mean, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(stats['var'], var, rtol=1e-13, atol=1e-12)
    np.testing.assert_allclose(stats['scale'], scale, rtol=1e-13, atol=1e-13)
    expected = ((content - mean) / scale).astype(np.float32)
    np.testing.assert_allclose(StandardizedContent(content, stats)[:], expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('chunk_size', [1, 7, 64, 1024])
def test_population_moments_and_transform_match_sklearn(chunk_size):
    StandardScaler = pytest.importorskip('sklearn.preprocessing').StandardScaler
    content, seen = fixture()
    expected = StandardScaler().fit(content[seen])
    stats = fit_standardization(content, seen, chunk_size=chunk_size, content_sha256='a' * 64)
    actual = StandardizedContent(content, stats, expected_content_sha256='a' * 64)
    np.testing.assert_allclose(actual.mean_, expected.mean_, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(actual.var_, expected.var_, rtol=1e-13, atol=1e-12)
    np.testing.assert_allclose(actual.scale_, expected.scale_, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(actual[:], expected.transform(content).astype(np.float32), rtol=1e-6, atol=1e-7)
    assert actual.scale_[-1] == 1. and actual[:][:, -1].tolist() == [0.] * len(content)
    assert actual.shape == content.shape and actual.ndim == 2 and len(actual) == len(content)
    assert actual.dtype == np.dtype('float32')
    assert stats['n_samples_seen'] == len(seen) and stats['ddof'] == 0
    assert stats['seen_rows_sha256'] == hashlib.sha256(seen.astype('<i8').tobytes()).hexdigest()


def test_chunk_invariance_is_numeric_not_a_bitwise_hash_promise():
    values = np.random.default_rng(55).normal(1.e8, 1.e2, size=(111, 5))
    rows = np.arange(len(values))
    expected_mean, expected_var = values.mean(axis=0), values.var(axis=0)
    variants = [fit_standardization(values, rows, chunk_size=c) for c in (1, 3, 13, 500)]
    for stats in variants:
        np.testing.assert_allclose(stats['mean'], expected_mean, rtol=0, atol=1.e-7)
        np.testing.assert_allclose(stats['var'], expected_var, rtol=1.e-9, atol=1.e-7)
        assert stats['seen_rows_sha256'] == variants[0]['seen_rows_sha256']


def test_near_constant_large_mean_uses_sklearn_constant_feature_guard():
    StandardScaler = pytest.importorskip('sklearn.preprocessing').StandardScaler
    values = np.array([[1.e15, 4.], [1.e15 + .125, 4.], [1.e15 - .125, 4.]])
    expected = StandardScaler().fit(values)
    stats = fit_standardization(values, np.arange(3), chunk_size=3)
    np.testing.assert_array_equal(stats['scale'], expected.scale_)
    assert stats['scale'] == [1., 1.]


def test_only_train_seen_rows_contribute_and_heldout_changes_do_not_change_fit():
    content, seen = fixture()
    untouched = fit_standardization(content, seen, chunk_size=7)
    changed = content.copy()
    heldout = np.setdiff1d(np.arange(len(content)), seen)
    changed[heldout] = 127
    assert fit_standardization(changed, seen, chunk_size=7) == untouched
    assert not np.allclose(np.mean(changed, axis=0), untouched['mean'])
    class SeenOnly:
        shape, ndim, dtype = content.shape, 2, content.dtype
        def __len__(self): return len(content)
        def __getitem__(self, rows):
            assert np.isin(rows, seen).all(), 'Fitter accessed a held-out row'
            return content[rows]
    assert fit_standardization(SeenOnly(), seen, chunk_size=7) == untouched


def test_nonfinite_cold_rows_are_not_read_at_fit_but_rejected_at_transform():
    content = np.array([[1., 3.], [2., 4.], [np.nan, np.inf]])
    stats = fit_standardization(content, np.array([0, 1]))
    wrapper = StandardizedContent(content, stats)
    assert np.isfinite(wrapper[[0, 1]]).all()
    with pytest.raises(ValueError, match='Nonfinite'):
        wrapper[2]


def test_single_training_item_and_zero_variance():
    content = np.array([[4., 0.], [7., 2.]])
    stats = fit_standardization(content, np.array([0]))
    assert stats['var'] == [0., 0.] and stats['scale'] == [1., 1.]
    np.testing.assert_array_equal(StandardizedContent(content, stats)[:], [[0., 0.], [3., 2.]])


def test_serialization_binding_tamper_detection_and_no_mutable_json_alias():
    content, seen = fixture()
    stats = fit_standardization(content, seen, content_sha256='b' * 64)
    reloaded = json.loads(json.dumps(stats))
    assert statistics_digest(reloaded) == stats['stats_sha256']
    wrapper = StandardizedContent(content, reloaded, expected_content_sha256='b' * 64)
    before = wrapper[0].copy()
    reloaded['mean'][0] += 10
    np.testing.assert_array_equal(wrapper[0], before)
    with pytest.raises(ValueError, match='SHA256'):
        validate_standardization(reloaded)
    with pytest.raises(ValueError, match='SHA256'):
        StandardizedContent(content, stats, expected_content_sha256='c' * 64)
    with pytest.raises(ValueError, match='shape or dtype'):
        StandardizedContent(content.astype(np.float32), stats)
    with pytest.raises(ValueError, match='shape or dtype'):
        StandardizedContent(content[:-1], stats)
    with pytest.raises(ValueError):
        wrapper.mean_[0] = 17


@pytest.mark.parametrize('rows', [[], [2, 1], [1, 1], [-1, 2], [0, 257], [0., 1.], [False, True], [[0, 1]]])
def test_invalid_fit_rows_rejected(rows):
    with pytest.raises(ValueError, match='train-seen rows'):
        fit_standardization(fixture()[0], rows)


@pytest.mark.parametrize('chunk', [0, -1, 1.5, True])
def test_invalid_chunks_rejected(chunk):
    with pytest.raises(ValueError, match='chunk_size'):
        fit_standardization(*fixture(), chunk_size=chunk)


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_train_content_rejected(value):
    values = np.array([[1., 2.], [value, 3.]])
    with pytest.raises(ValueError, match='Nonfinite'):
        fit_standardization(values, np.arange(2))


@pytest.mark.parametrize('content', [np.ones((2, 2), dtype=bool), np.zeros((2, 0)),
    np.ones((2, 2), dtype=complex), np.ones(2), np.full((2, 2), '2')])
def test_invalid_content_matrix_rejected(content):
    with pytest.raises(ValueError, match='content matrix'):
        fit_standardization(content, np.arange(2))


@pytest.mark.parametrize('selection', [-1, 257, [-1, 0], [257], [1.], [True], True,
    (0, 1), slice(-1, None), slice(0, 258), slice(None, None, -1), slice(None, None, 0)])
def test_invalid_transform_row_selection_rejected(selection):
    content, seen = fixture()
    wrapper = StandardizedContent(content, fit_standardization(content, seen))
    with pytest.raises(ValueError):
        wrapper[selection]


def test_valid_row_index_shapes_and_raw_asset_unchanged():
    content, seen = fixture()
    before = content.copy()
    wrapper = StandardizedContent(content, fit_standardization(content, seen))
    assert wrapper[0].shape == (9,)
    assert wrapper[[0, 0, 2]].shape == (3, 9)
    assert wrapper[np.array([[0, 1], [2, 3]])].shape == (2, 2, 9)
    assert wrapper[np.array([], dtype=np.int64)].shape == (0, 9)
    assert wrapper[0:0].shape == (0, 9)
    np.testing.assert_array_equal(wrapper[2:10:2], wrapper[[2, 4, 6, 8]])
    np.testing.assert_array_equal(content, before)


@pytest.mark.parametrize('field,value', [
    ('extra', 1), ('schema', 'wrong'), ('ddof', 1), ('ddof', False),
    ('fit_scope', 'all_items'), ('n_samples_seen', 0), ('n_samples_seen', 258),
    ('content_shape', [257, 0]), ('content_dtype', 'int8'), ('content_sha256', 'x'),
    ('seen_rows_sha256', 'f'), ('mean', [0.]), ('var', [-1.] * 9), ('scale', [0.] * 9),
    ('scale', [999.] * 9), ('mean', [np.nan] * 9), ('mean', [True] * 9),
])
def test_invalid_statistics_schema_rejected_even_if_redigested(field, value):
    content, seen = fixture()
    stats = copy.deepcopy(fit_standardization(content, seen))
    stats[field] = value
    with pytest.raises(ValueError):
        stats['stats_sha256'] = statistics_digest(stats)
        validate_standardization(stats)


def test_history_training_and_cold_paths_share_one_transform_and_padding_stays_zero():
    import torch
    import torch.nn.functional as F
    from src.baselines.liger_full_catalog import RowCatalog, full_catalog_cross_entropy
    from src.baselines.liger_inputs import pack_sequences
    from src.baselines.liger_retrieval import ColdCatalogCache, cold_rows_from_seen

    content, seen = fixture()
    sid = np.arange(len(content) * 4).reshape(-1, 4) % 64
    stats = fit_standardization(content, seen, chunk_size=17)
    standardized = StandardizedContent(content, stats)
    raw_seen = content[seen].astype(np.float64)
    expected_scale = raw_seen.std(axis=0)
    expected_scale[expected_scale == 0] = 1.
    expected = ((content - raw_seen.mean(axis=0)) / expected_scale).astype(np.float32)
    batch = pack_sequences([[0, 3, 6], [3, 9]], sid, standardized, codebook_size=64)
    np.testing.assert_allclose(batch['input_embeddings'][0, :2], expected[[0, 3]])
    np.testing.assert_allclose(batch['input_embeddings'][1, :1], expected[[3]])
    assert np.count_nonzero(batch['input_embeddings'][0, 2:]) == 0
    assert np.count_nonzero(batch['input_embeddings'][1, 1:]) == 0
    # Real asset row zero is transformed, not silently converted to padding.
    assert np.count_nonzero(batch['input_embeddings'][0, 0]) > 0
    catalog = RowCatalog(standardized, seen)
    np.testing.assert_allclose(catalog[:], expected[seen])

    torch.manual_seed(9)
    projection = torch.nn.Linear(content.shape[1], 5).eval()
    cold = cold_rows_from_seen(len(content), seen)
    cache = ColdCatalogCache(projection, standardized, cold, device='cpu', chunk_size=13)
    with torch.no_grad():
        oracle_vectors = F.normalize(projection(torch.from_numpy(expected[cold])), dim=-1)
    torch.testing.assert_close(cache.vectors, oracle_vectors, rtol=1e-5, atol=1e-6)
    query = torch.randn(2, 5, requires_grad=True)
    labels = torch.tensor([0, 3])
    actual_loss = full_catalog_cross_entropy(query, labels, projection, catalog, chunk_size=13)
    oracle_logits = F.normalize(query, dim=-1) @ F.normalize(projection(torch.from_numpy(expected[seen])), dim=-1).T / .07
    oracle_loss = F.cross_entropy(oracle_logits, labels)
    torch.testing.assert_close(actual_loss, oracle_loss, rtol=1e-5, atol=1e-6)
    for actual, oracle in zip(torch.autograd.grad(actual_loss, (query, *projection.parameters())),
                              torch.autograd.grad(oracle_loss, (query, *projection.parameters()))):
        torch.testing.assert_close(actual, oracle, rtol=1e-4, atol=1e-5)
