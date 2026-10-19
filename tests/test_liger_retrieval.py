import inspect

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.baselines.liger_retrieval import (
    ColdCatalogCache, cold_rows_from_seen, exact_topk, full_cold_rank, ranking_metrics,
)
from src.baselines.liger_candidates import rank_unique_rows


@pytest.mark.parametrize('k', [1, 4, 10])
def test_exact_topk_resolves_boundary_ties_without_score_perturbation(k):
    rows = torch.tensor([90, 2, 51, 4, 13, 20, 11, 43, 7, 9])
    scores = torch.tensor([[0.] * 10, [2., 3., 1., 2., 3., 2., 2., 0., 1., 2.]])
    actual_scores, actual_rows = exact_topk(scores, rows, k)
    for i in range(2):
        expected_rows, expected_scores = rank_unique_rows(rows.numpy(), scores[i].numpy(), k)
        np.testing.assert_array_equal(actual_rows[i].numpy(), expected_rows)
        np.testing.assert_array_equal(actual_scores[i].numpy(), expected_scores)


def fixture(chunk_size=7):
    torch.manual_seed(42)
    projection = torch.nn.Linear(9, 4).eval()
    content = np.random.default_rng(42).normal(size=(61, 9)).astype(np.float32)
    seen = np.array([0, 2, 3, 7, 11], dtype=np.int32)
    cold = cold_rows_from_seen(len(content), seen)
    cache = ColdCatalogCache(projection, content, cold, device='cpu', chunk_size=chunk_size)
    query = torch.randn(2, 4)
    query[1].zero_()  # Every score ties: ranking must use global row identity.
    return projection, content, seen, cold, cache, query


@pytest.mark.parametrize('chunk_size', [3, 7, 64])
def test_complete_cold_union_matches_monolithic_oracle_and_retains_row_zero(chunk_size):
    projection, content, seen, cold, cache, query = fixture(chunk_size)
    beams = np.array([[0, 2, 8, -1, 2], [0, 7, 3, 3, -1]])
    actual = full_cold_rank(cache, query, beams)
    with torch.no_grad():
        all_vectors = F.normalize(projection(torch.from_numpy(content)), dim=-1)
        dense = (F.normalize(query, dim=-1) @ all_vectors.T / .07).numpy()
    for i in range(2):
        union = np.unique(np.concatenate([cold, beams[i][beams[i] >= 0]]))
        rows, scores = rank_unique_rows(union, dense[i, union], 20)
        np.testing.assert_array_equal(actual['hybrid_rows'][i], rows)
        np.testing.assert_allclose(actual['hybrid_scores'][i], scores, rtol=1e-5, atol=3e-6)
    assert actual['hybrid_rows'][1, 0] == 0
    assert actual['valid_beam_rows'][0].tolist() == [0, 2, 8, -1, -1]
    assert actual['beam_only_rows'][1, :3].tolist() == [0, 3, 7]
    assert not {'target', 'targets', 'labels', 'gt'} & set(inspect.signature(full_cold_rank).parameters)


def test_all_invalid_beams_still_retrieve_entire_cold_pool():
    _, _, _, cold, cache, query = fixture()
    out = full_cold_rank(cache, query, np.full((2, 20), -1, np.int32))
    assert (out['beam_only_rows'] == -1).all()
    assert np.isin(out['hybrid_rows'], cold).all()
    assert len(np.unique(out['hybrid_rows'][0])) == 20


@pytest.mark.parametrize('change', ['training', 'weights', 'autocast'])
def test_cache_cannot_silently_outlive_projection_state(change):
    projection, _, _, _, cache, query = fixture()
    if change == 'training': projection.train()
    if change == 'weights':
        with torch.no_grad(): projection.weight.add_(1)
    if change == 'autocast':
        with torch.autocast('cpu', dtype=torch.bfloat16), pytest.raises(RuntimeError):
            cache.topk(query)
    else:
        with pytest.raises(RuntimeError): cache.topk(query)


def test_metrics_at_5_10_20_count_all_users_and_single_correct_item():
    rows = np.tile(np.arange(20), (4, 1))
    result = ranking_metrics(rows, np.array([0, 5, 19, 77]))
    assert result['Recall@5'] == .25 and result['Recall@10'] == .5 and result['Recall@20'] == .75
    assert result['NDCG@20'] == pytest.approx((1 + 1 / np.log2(7) + 1 / np.log2(21)) / 4)
    rows[0, 1] = 0
    with pytest.raises(ValueError, match='unique'): ranking_metrics(rows, np.array([0, 5, 19, 77]))


def test_cold_complement_is_based_only_on_declared_train_seen_rows():
    assert cold_rows_from_seen(6, np.array([0, 2, 5])).tolist() == [1, 3, 4]
    for bad in ([2, 0], [0, 0], [-1, 2], [0, 6]):
        with pytest.raises(ValueError): cold_rows_from_seen(6, np.array(bad))
