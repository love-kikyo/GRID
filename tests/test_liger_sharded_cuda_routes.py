"""Real two-4090 routing checks, using synthetic items and no data labels.

These are integration/path tests, not recommendation-quality measurements.
They deliberately create valid seen and cold beams that an almost-untrained
checkpoint's natural decoder may not produce. The projection is taken from
the actual fresh seed42 384-wide model factory, without loading a checkpoint.
CPU-only test runs skip; the GPU-only launcher must reject an all-skipped run.
"""
from pathlib import Path
from types import SimpleNamespace
import os

import numpy as np
import pytest
import torch

from src.baselines.liger_capacity import CAPACITY384, build_capacity_model
from src.baselines.liger_retrieval import ColdCatalogCache, full_cold_rank
from src.baselines.liger_sharded_retrieval import ShardedColdCatalogCache


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason='Synthetic route integration requires two allocated CUDA GPUs',
)
ROOT = Path(__file__).resolve().parents[1]
SCORE_ATOL = 1e-5
SCORE_RTOL = 1e-5


@pytest.fixture(scope='module')
def cuda_routes():
    assert torch.cuda.device_count() == 2, 'Allocate exactly two GPUs for this test'
    assert all('4090' in torch.cuda.get_device_name(i) for i in range(2))
    old_tf32 = (torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sharded = None
    try:
        with torch.random.fork_rng(devices=[0, 1]):
            model, _, _, report = build_capacity_model(
                ROOT.parent / 'liger-upstream', CAPACITY384, seed=42, device='cpu')
            assert report['disjoint_groups']['emb_proj'] == 1348480
            projection = model.eval().emb_proj.to('cuda:0')
            del model
            before = {key: value.detach().cpu().clone()
                      for key, value in projection.state_dict().items()}
            content = np.random.default_rng(42).normal(size=(96, 128)).astype(np.float32)
            cold = np.arange(1, 96, 2, dtype=np.int32)
            # Both replicas and the mono reference project/scan 24-row chunks.
            # This avoids attributing GEMM-shape rounding to cross-device routing.
            sharded = ShardedColdCatalogCache(
                projection, content, cold, devices=('cuda:0', 'cuda:1'), chunk_size=24)
            mono = ColdCatalogCache(projection, content, cold,
                                    device='cuda:0', chunk_size=24)
            queries = torch.as_tensor(
                np.random.default_rng(43).normal(size=(5, 384)).astype(np.float32),
                device='cuda:0')
            with torch.no_grad():
                queries[0] = mono.vectors[0]   # Primary-shard cold row 1.
                queries[1] = mono.vectors[24]  # Secondary-shard cold row 49.
                queries[2] = mono._project(np.array([0], dtype=np.int32))[0]
                queries[3].zero_()             # Exact ties across both devices.
            beams = np.array([
                [1, 49, 0, 2, 49, -1, 95, 47],
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [0, 2, 94, 0, -1, 94, -1, -1],
                [95, 47, 0, 2, 95, -1, 1, 49],
                [49, 1, 94, 2, 1, -1, 95, 47],
            ], dtype=np.int32)
            yield SimpleNamespace(projection=projection, content=content, cold=cold,
                                  sharded=sharded, mono=mono, queries=queries, beams=beams)
            # Every tested inference route must leave both copies unchanged.
            for shard in sharded.shards:
                assert not any(module.training for module in shard.projection.modules())
                for key, value in shard.projection.state_dict().items():
                    torch.testing.assert_close(value.detach().cpu(), before[key], atol=0, rtol=0)
    finally:
        if sharded is not None:
            sharded.close()
        (torch.backends.cuda.matmul.allow_tf32,
         torch.backends.cudnn.allow_tf32) = old_tf32


def assert_rank_equal(actual, expected):
    for key in ('hybrid_rows', 'beam_only_rows', 'valid_beam_rows'):
        np.testing.assert_array_equal(actual[key], expected[key])
    for key in ('hybrid_scores', 'beam_only_scores', 'beam_scores'):
        np.testing.assert_allclose(actual[key], expected[key],
                                   atol=SCORE_ATOL, rtol=SCORE_RTOL)
    for rows in actual['hybrid_rows']:
        valid = rows[rows >= 0]
        assert len(valid) == len(np.unique(valid))


@pytest.mark.parametrize('k', [1, 20, 48])
def test_cuda_exact_global_topk_and_complete_partition(cuda_routes, k):
    f = cuda_routes
    rows, scores = f.sharded.topk(f.queries, k=k)
    ref_rows, ref_scores = f.mono.topk(f.queries, k=k)
    torch.testing.assert_close(rows, ref_rows, atol=0, rtol=0)
    torch.testing.assert_close(scores, ref_scores, atol=SCORE_ATOL, rtol=SCORE_RTOL)
    assert rows.device == scores.device == torch.device('cuda:0')
    assert f.sharded.shard_sizes == (24, 24)
    assert [shard.last_scan_rows for shard in f.sharded.shards] == [24, 24]
    assert f.sharded.last_scan_rows == f.mono.last_scan_rows == len(f.cold)
    np.testing.assert_array_equal(np.concatenate(f.sharded.partition_rows), f.cold)
    assert f.sharded.cache_bytes == (24 * (384 * 4 + 8),) * 2
    assert rows[0, 0].item() == 1 and rows[1, 0].item() == 49
    # The zero query creates exact cutoff ties; the global row ID decides.
    np.testing.assert_array_equal(rows[3].cpu().numpy(), f.cold[:k])
    assert not scores.requires_grad


def test_cuda_cold_gathers_both_devices_and_seen_projection_only(cuda_routes, monkeypatch):
    f = cuda_routes
    assert [shard.vectors.device for shard in f.sharded.shards] == [
        torch.device('cuda:0'), torch.device('cuda:1')]
    assert all(shard.vectors.shape == (24, 384) for shard in f.sharded.shards)
    projected_rows = []
    original = f.sharded.shards[0]._project

    def primary_project(rows):
        projected_rows.append(np.array(rows, copy=True))
        return original(rows)

    def no_secondary_projection(_):
        pytest.fail('Cold beam vectors must be gathered from the existing secondary cache')

    monkeypatch.setattr(f.sharded.shards[0], '_project', primary_project)
    monkeypatch.setattr(f.sharded.shards[1], '_project', no_secondary_projection)
    with torch.enable_grad():
        rows, scores = f.sharded.score_beams(f.queries.detach().requires_grad_(), f.beams)
    ref_rows, ref_scores = f.mono.score_beams(f.queries, f.beams)
    np.testing.assert_array_equal(rows, ref_rows)
    torch.testing.assert_close(scores, ref_scores, atol=SCORE_ATOL, rtol=SCORE_RTOL)
    # Independent sanitation expectations include repeated row 0 (a valid item).
    assert rows[0].tolist() == [1, 49, 0, 2, -1, -1, 95, 47]
    assert rows[2].tolist() == [0, 2, 94, -1, -1, -1, -1, -1]
    flat = rows[rows >= 0]
    assert (flat == 1).any() and (flat == 47).any()
    assert (flat == 49).any() and (flat == 95).any()
    assert len(projected_rows) == 1
    np.testing.assert_array_equal(projected_rows[0], flat[flat % 2 == 0])
    assert torch.isneginf(scores[torch.as_tensor(rows < 0, device='cuda:0')]).all()
    assert not scores.requires_grad


def test_cuda_mixed_empty_rows_and_seen_only_beams_match_mono(cuda_routes):
    f = cuda_routes
    actual = f.sharded.rank(f.queries, f.beams, k=20)
    assert_rank_equal(actual, full_cold_rank(f.mono, f.queries, f.beams, k=20))
    assert (actual['beam_only_rows'][1] == -1).all()
    assert np.isneginf(actual['beam_only_scores'][1]).all()
    assert actual['hybrid_rows'][2, 0] == 0
    expected_tied = np.sort(np.concatenate([f.cold, [0, 2]]))[:20]
    np.testing.assert_array_equal(actual['hybrid_rows'][3], expected_tied)
    np.testing.assert_array_equal(actual['beam_only_rows'][3, :6], [0, 1, 2, 47, 49, 95])


def test_cuda_zero_width_beam_matches_cold_only(cuda_routes):
    f = cuda_routes
    beams = np.empty((len(f.queries), 0), dtype=np.int32)
    actual = f.sharded.rank(f.queries, beams)
    assert_rank_equal(actual, full_cold_rank(f.mono, f.queries, beams))
    cold_rows, cold_scores = f.mono.topk(f.queries)
    np.testing.assert_array_equal(actual['hybrid_rows'], cold_rows.cpu().numpy())
    np.testing.assert_allclose(actual['hybrid_scores'], cold_scores.cpu().numpy(),
                               atol=SCORE_ATOL, rtol=SCORE_RTOL)
    assert actual['valid_beam_rows'].shape == actual['beam_scores'].shape == (5, 0)
    assert (actual['beam_only_rows'] == -1).all()


def test_cuda_global_cold_exclusion_ignores_beam_rounding_on_both_shards(cuda_routes, monkeypatch):
    f = cuda_routes
    query = torch.zeros((1, 384), device='cuda:0')
    # Both are cold but outside the exact zero-query top20; one on each GPU.
    beams = np.array([[47, 95, 47, -1]], dtype=np.int32)
    original = f.sharded.score_beams

    def exaggerated_beam_scores(*args, **kwargs):
        rows, scores = original(*args, **kwargs)
        scores[torch.as_tensor(rows >= 0, device='cuda:0')] = 1000.
        return rows, scores

    monkeypatch.setattr(f.sharded, 'score_beams', exaggerated_beam_scores)
    actual = f.sharded.rank(query, beams)
    # A global-cold exclusion bug would incorrectly promote row47 or row95.
    np.testing.assert_array_equal(actual['hybrid_rows'][0], f.cold[:20])
    np.testing.assert_array_equal(actual['hybrid_scores'][0], np.zeros(20))
    np.testing.assert_array_equal(actual['beam_only_rows'][0, :2], [47, 95])
    np.testing.assert_array_equal(actual['beam_only_scores'][0, :2], [1000., 1000.])


def test_cuda_dispatch_threads_stay_within_four_cpu_affinity(cuda_routes):
    from src.runtime_export_cpu import export_cpu_snapshot

    f = cuda_routes
    f.sharded.topk(f.queries)
    executor = f.sharded._executor
    assert executor is not None and executor._threads
    assert all(thread.is_alive() for thread in executor._threads)
    allowed = sorted(os.sched_getaffinity(0))
    snapshot = export_cpu_snapshot(allowed)
    assert len(allowed) <= 4 and snapshot['processes']
    print('active CUDA dispatcher CPU affinity:', snapshot)


@pytest.mark.parametrize('bad_row', [-2, 96])
def test_cuda_invalid_beam_rows_rejected(cuda_routes, bad_row):
    f = cuda_routes
    bad = f.beams.copy()
    bad[0, 0] = bad_row
    with pytest.raises(ValueError):
        f.sharded.score_beams(f.queries, bad)
