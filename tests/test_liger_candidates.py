import inspect

import numpy as np
import pytest

from src.baselines.liger_candidates import SidRowIndex, sid_keys, unique_beam_rows, rank_unique_rows


def fixture():
    return np.array([[0, 0, 0, 0], [9, 2, 4, 6], [1, 0, 0, 0], [4095] * 4], dtype=np.int32)


def test_index_roundtrip_arbitrary_order_and_maximum_code():
    sids = fixture()
    index = SidRowIndex.build(sids, chunk_size=2)
    assert index.lookup_codes(sids).tolist() == [0, 1, 2, 3]
    assert index.rows.tolist() == [0, 2, 1, 3]
    assert sid_keys(sids)[-1] == 2**48 - 1
    assert index.lookup_codes(np.array([[0, 0, 0, 1]])).tolist() == [-1]


def test_unknown_beyond_largest_key_is_missing_not_last_or_zero():
    index = SidRowIndex.build(fixture()[:-1])
    assert index.lookup_codes(fixture()[-1:]).tolist() == [-1]


def test_padding_eos_wrong_level_and_unknown_sid_never_alias_row_zero():
    index = SidRowIndex.build(fixture())
    valid_zero = np.array([1, 4097, 8193, 12289])
    tokens = np.stack([valid_zero, [0, 0, 0, 0], [16385, 0, 0, 0],
                      [4097, 4097, 8193, 12289], valid_zero + [0, 0, 0, 1]])
    assert index.lookup_tokens(tokens).tolist() == [0, -1, -1, -1, -1]


@pytest.mark.parametrize('bad', [np.array([[0., 1, 2, 3]]), np.array([[-1, 0, 0, 0]]),
                               np.array([[4096, 0, 0, 0]]), np.array([[0, 1, 2]])])
def test_invalid_sid_codes_are_rejected(bad):
    with pytest.raises(ValueError):
        sid_keys(bad)


def test_aliases_and_nonbijective_mapping_are_not_silently_resolved():
    with pytest.raises(ValueError, match='aliases'):
        SidRowIndex.build(np.zeros((2, 4), dtype=np.int32))
    with pytest.raises(ValueError, match='exactly once'):
        SidRowIndex(np.array([0, 1], dtype=np.uint64), np.array([0, 0], dtype=np.int32))


def test_beam_sanitation_preserves_slots_and_does_not_backfill():
    rows = np.array([[0, -1, 0, 3, 3], [2, 1, -1, -1, 0]])
    assert unique_beam_rows(rows).tolist() == [[0, -1, -1, 3, -1], [2, 1, -1, -1, 0]]
    assert rows[0, 2] == 0  # input is unchanged


def test_dense_ranking_deduplicates_and_has_deterministic_ties():
    rows, scores = rank_unique_rows([3, 0, 3, -1, 2], [.5, .8, .9, np.nan, .8], 4)
    assert rows.tolist() == [3, 0, 2, -1]
    np.testing.assert_allclose(scores, [.9, .8, .8, -np.inf])
    assert 'label' not in inspect.signature(rank_unique_rows).parameters


def test_chunk_topk_merge_matches_full_unique_dense_ranking():
    rows = np.arange(31)
    scores = np.random.default_rng(4).random(31)
    parts = [rank_unique_rows(rows[i:i + 7], scores[i:i + 7], 5) for i in range(0, 31, 7)]
    merged = rank_unique_rows(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), 5)
    full = rank_unique_rows(rows, scores, 5)
    np.testing.assert_array_equal(merged[0], full[0])
    np.testing.assert_array_equal(merged[1], full[1])


def test_nonfinite_valid_score_and_invalid_missing_sentinel_rejected():
    with pytest.raises(ValueError):
        rank_unique_rows([0], [np.nan], 1)
    with pytest.raises(ValueError):
        rank_unique_rows([-2], [0.], 1)
    with pytest.raises(ValueError):
        rank_unique_rows([2**32], [0.], 1)
