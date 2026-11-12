import inspect
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from src.baselines.liger_retrieval import ColdCatalogCache, cold_rows_from_seen, full_cold_rank
from src.baselines.liger_sharded_retrieval import ShardedColdCatalogCache, _scan


def fixture(chunk_size=7):
    torch.manual_seed(42)
    projection = torch.nn.Linear(9, 4).eval()
    content = np.random.default_rng(42).normal(size=(62, 9)).astype(np.float32)
    seen = np.array([0, 2, 3, 7, 11], dtype=np.int32)
    cold = cold_rows_from_seen(len(content), seen)
    cache = ShardedColdCatalogCache(projection, content, cold,
                                   devices=('cpu', 'cpu'), chunk_size=chunk_size)
    query = torch.randn(3, 4)
    query[1].zero_()
    beams = np.array([[0, 2, 8, 61, 2], [0, 7, 3, 3, -1], [-1] * 5])
    return projection, content, cold, cache, query, beams


@pytest.mark.parametrize('chunk_size', [1, 3, 7, 64])
@pytest.mark.parametrize('k', [1, 5, 20, 57])
def test_global_topk_exact_rows_and_scores_against_unsharded_cache(chunk_size, k):
    projection, content, cold, cache, query, _ = fixture(chunk_size)
    reference = ColdCatalogCache(projection, content, cold, device='cpu', chunk_size=chunk_size)
    rows, scores = cache.topk(query, k)
    expected_rows, expected_scores = reference.topk(query, k)
    torch.testing.assert_close(rows, expected_rows, atol=0, rtol=0)
    torch.testing.assert_close(scores, expected_scores, atol=3e-6, rtol=1e-5)
    assert cache.last_scan_rows == len(cold)
    assert sum(shard.last_scan_rows for shard in cache.shards) == len(cold)
    assert cache.shard_sizes == (29, 28)
    assert cache.cache_bytes == (29 * (4 * 4 + 8), 28 * (4 * 4 + 8))


@pytest.mark.parametrize('chunk_size', [3, 7, 64])
def test_rank_complete_union_matches_reference_including_secondary_cold_beams(chunk_size):
    projection, content, cold, cache, query, beams = fixture(chunk_size)
    reference = ColdCatalogCache(projection, content, cold, device='cpu', chunk_size=chunk_size)
    actual = cache.rank(query, beams)
    expected = full_cold_rank(reference, query, beams)
    for key in ('hybrid_rows', 'beam_only_rows', 'valid_beam_rows'):
        np.testing.assert_array_equal(actual[key], expected[key])
    for key in ('hybrid_scores', 'beam_only_scores', 'beam_scores'):
        np.testing.assert_allclose(actual[key], expected[key], atol=3e-6, rtol=1e-5)
    assert actual['hybrid_rows'][1, 0] == 0
    assert actual['valid_beam_rows'][0].tolist() == [0, 2, 8, 61, -1]
    assert (actual['beam_only_rows'][2] == -1).all()
    assert not {'target', 'targets', 'labels', 'gt'} & set(inspect.signature(cache.rank).parameters)


def test_boundary_ties_across_shards_obey_global_row_order():
    projection = torch.nn.Linear(3, 2, bias=False).eval()
    content = np.ones((45, 3), dtype=np.float32)
    cold = np.arange(1, 45, dtype=np.int32)
    with ShardedColdCatalogCache(projection, content, cold, devices=('cpu', 'cpu'), chunk_size=7) as cache:
        query = torch.zeros(2, 2)
        out = cache.rank(query, np.array([[0, 43, -1], [-1, -1, -1]]))
        np.testing.assert_array_equal(out['hybrid_rows'][0], np.arange(20))
        np.testing.assert_array_equal(out['hybrid_rows'][1], np.arange(1, 21))


def test_cold_beam_rounding_cannot_override_global_scan_on_either_shard(monkeypatch):
    _, _, cold, cache, query, _ = fixture()
    query.zero_()
    original = cache.score_beams
    def changed_scores(q, rows, temperature=.07):
        beam_rows, scores = original(q, rows, temperature)
        scores[beam_rows >= 0] = 1000.
        return beam_rows, scores
    monkeypatch.setattr(cache, 'score_beams', changed_scores)
    out = cache.rank(query, np.array([[60, 61], [60, 61], [60, 61]]))
    for row in out['hybrid_rows']:
        np.testing.assert_array_equal(row, cold[:20])


@pytest.mark.parametrize('index', [0, 1])
@pytest.mark.parametrize('change', ['training', 'weights', 'parameter_replace'])
def test_all_shards_reject_stale_projection_or_training_mode(index, change):
    _, _, _, cache, query, beams = fixture()
    projection = cache.shards[index].projection
    if change == 'training':
        projection.train()
    elif change == 'weights':
        with torch.no_grad(): projection.weight.add_(1)
    else:
        projection.weight = torch.nn.Parameter(projection.weight.detach().clone())
    for action in (lambda: cache.topk(query), lambda: cache.score_beams(query, beams),
                   lambda: cache.rank(query, beams)):
        with pytest.raises(RuntimeError): action()


def test_secondary_projection_is_an_independent_exact_copy():
    projection, _, _, cache, _, _ = fixture()
    copied = cache.shards[1].projection
    assert copied is not projection
    for first, second in zip(projection.parameters(), copied.parameters()):
        assert first is not second
        torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_input_catalog_identities_are_owned_and_immutable():
    _, _, cold, cache, _, _ = fixture()
    original = cache.rows_np.copy()
    cold[:] = 0
    np.testing.assert_array_equal(cache.rows_np, original)
    assert not cache.rows_np.flags.writeable
    assert all(not rows.flags.writeable for rows in cache.partition_rows)
    assert np.concatenate(cache.partition_rows).tolist() == original.tolist()


def test_incomplete_scan_is_rejected(monkeypatch):
    _, _, _, cache, query, _ = fixture()
    shard = cache.shards[1]
    original = shard.topk
    def partial(*args):
        result = original(*args)
        shard.last_scan_rows -= 1
        return result
    monkeypatch.setattr(shard, 'topk', partial)
    with pytest.raises(RuntimeError, match='Incomplete'): cache.topk(query)


@pytest.mark.parametrize('change', ['rows', 'device', 'shard'])
def test_partition_replacement_is_rejected(change):
    _, _, _, cache, query, _ = fixture()
    if change == 'rows': cache.shards[1].rows_np = cache.shards[1].rows_np.copy()
    if change == 'device': cache.shards[1].device = torch.device('cpu:1')
    if change == 'shard': cache.shards = (cache.shards[0], cache.shards[0])
    with pytest.raises(RuntimeError, match='shard'): cache.topk(query)


def test_autocast_and_closed_cache_are_rejected():
    _, _, _, cache, query, beams = fixture()
    with torch.autocast('cpu', dtype=torch.bfloat16), pytest.raises(RuntimeError):
        cache.rank(query, beams)
    cache.close()
    cache.close()
    with pytest.raises(RuntimeError, match='closed'): cache.topk(query)


def test_worker_always_runs_without_autograd(monkeypatch):
    _, _, _, cache, query, _ = fixture()
    observed = []
    original = cache.shards[0].topk
    def spy(*args):
        observed.append(torch.is_grad_enabled())
        return original(*args)
    monkeypatch.setattr(cache.shards[0], 'topk', spy)
    with torch.enable_grad(): _scan(cache.shards[0], query.requires_grad_(), 3, .07)
    assert observed == [False]


def test_threaded_dispatch_uses_both_shards_and_closes_workers():
    _, _, _, cache, query, _ = fixture()
    expected = cache.topk(query, 20)
    # Exercise the dispatcher without requiring GPUs in the unit-test process.
    cache._executor = ThreadPoolExecutor(max_workers=2)
    actual = cache.topk(query.requires_grad_(), 20)
    for first, second in zip(actual, expected):
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        assert not first.requires_grad
    cache.close()
    assert cache._executor is None and cache._closed


def test_worker_failure_propagates_and_shuts_down_dispatcher(monkeypatch):
    _, _, _, cache, query, _ = fixture()
    cache._executor = ThreadPoolExecutor(max_workers=2)
    def fail(*_):
        raise RuntimeError('deliberate shard failure')
    monkeypatch.setattr(cache.shards[1], 'topk', fail)
    with pytest.raises(RuntimeError, match='deliberate shard failure'):
        cache.topk(query)
    assert cache._executor is None and cache._closed
    assert cache.last_scan_rows == 0


@pytest.mark.parametrize('devices', [(), ('cpu',), ('cpu', 'cpu', 'cpu'),
                                     ('cuda:0', 'cuda:0'), ('cpu', 'cuda:0'), ('meta', 'meta')])
def test_bad_device_partitions_fail_before_allocation(devices):
    with pytest.raises(ValueError):
        ShardedColdCatalogCache(torch.nn.Linear(2, 2).eval(), np.ones((4, 2)),
                                np.array([0, 1, 2]), devices=devices)


@pytest.mark.parametrize('rows', [[], [0], [-1, 0], [1, 1], [2, 1], [0, 4], [0., 1.]])
def test_bad_cold_rows_rejected(rows):
    with pytest.raises(ValueError):
        ShardedColdCatalogCache(torch.nn.Linear(2, 2).eval(), np.ones((4, 2)),
                                np.array(rows), devices=('cpu', 'cpu'))


@pytest.mark.parametrize('k', [0, -1, 58, True, 1.5])
def test_invalid_k_rejected(k):
    _, _, _, cache, query, _ = fixture()
    with pytest.raises(ValueError): cache.topk(query, k)


@pytest.mark.parametrize('temperature', [0, -1, np.nan, np.inf])
def test_invalid_temperature_rejected(temperature):
    _, _, _, cache, query, beams = fixture()
    with pytest.raises(ValueError): cache.topk(query, temperature=temperature)
    with pytest.raises(ValueError): cache.score_beams(query, beams, temperature=temperature)


def test_invalid_query_and_beams_rejected():
    _, _, _, cache, query, beams = fixture()
    for wrong in (query[:, :3], query[0], query * float('nan')):
        with pytest.raises(ValueError): cache.topk(wrong)
    for wrong in (beams[:1], np.full_like(beams, 62), np.full_like(beams, -2)):
        with pytest.raises(ValueError): cache.score_beams(query, wrong)
