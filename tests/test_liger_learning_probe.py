from dataclasses import replace

import numpy as np
import pytest

from src.baselines.liger_data import Record
from src.baselines.liger_learning_probe import (
    deterministic_batches, diagnostic_catalog, generation_metrics, split_parents,
)
from src.baselines.liger_supervision import expand_training_records


def parents(count=8):
    return [Record(i, (i, i + 1, i + 2), f'{i:064x}') for i in range(count)]


def test_split_is_exact_disjoint_deterministic_and_seeded():
    records = parents(256)
    fit, reserved = split_parents(records)
    assert len(fit) == len(reserved) == 128
    assert {r.user_id for r in fit}.isdisjoint(r.user_id for r in reserved)
    assert sorted(r.user_id for r in fit + reserved) == list(range(256))
    assert (fit, reserved) == split_parents(iter(records))
    assert (fit, reserved) != split_parents(records, seed=12)
    assert records == parents(256)


def test_split_rejects_wrong_counts_duplicate_users_and_bad_identity():
    with pytest.raises(ValueError, match='count'):
        split_parents(parents(7), fit_count=4, reserved_count=4)
    with pytest.raises(ValueError, match='Duplicate parent'):
        split_parents([parents(1)[0]] * 2, fit_count=1, reserved_count=1)
    with pytest.raises(ValueError, match='SHA256'):
        split_parents([Record(0, (0, 1), 'bad'), parents(2)[1]], fit_count=1, reserved_count=1)


def test_catalog_contains_bounded_fit_window_only_and_random_seen_fill():
    records = [Record(1, tuple(range(20)), 'a' * 64), Record(2, (30, 31, 30), 'b' * 64)]
    seen = np.arange(12, 200, dtype=np.int32)
    catalog = diagnostic_catalog(records, seen, size=24)
    assert len(catalog) == 24 and catalog.dtype == np.int64
    assert (np.diff(catalog) > 0).all()
    assert set(range(12, 20)) | {30, 31} <= set(catalog)
    assert set(catalog) <= set(seen)
    np.testing.assert_array_equal(catalog, diagnostic_catalog(records, seen, size=24))
    assert not np.array_equal(catalog, diagnostic_catalog(records, seen, size=24, seed=8))


def test_catalog_no_fill_and_entire_seen_catalog_edge_cases():
    record = Record(1, (0, 1, 2), 'a' * 64)
    np.testing.assert_array_equal(diagnostic_catalog([record], np.arange(3), size=3), np.arange(3))
    np.testing.assert_array_equal(diagnostic_catalog([record], np.arange(7), size=7), np.arange(7))


@pytest.mark.parametrize('seen', [np.array([0, 2, 1]), np.array([0, 1, 1, 2]),
                                  np.array([-1, 0, 1, 2]), np.array([0., 1., 2.]),
                                  np.array([[0, 1, 2]]), np.array([0, 1])])
def test_catalog_rejects_noncanonical_seen(seen):
    with pytest.raises(ValueError, match='seen_rows'):
        diagnostic_catalog(parents(1), seen, size=3)


@pytest.mark.parametrize('seen', [np.array([0, 1, 3, 4]), np.array([3, 4, 5, 6])])
def test_catalog_rejects_missing_required_seen_membership(seen):
    with pytest.raises(ValueError, match='belong'):
        diagnostic_catalog(parents(1), seen, size=3)


def test_fit_unique_rows_cannot_exceed_catalog_size():
    with pytest.raises(ValueError, match='exceed'):
        diagnostic_catalog(parents(2), np.arange(20), size=3)


def test_batches_are_reproducible_full_and_without_replacement_per_epoch():
    examples = expand_training_records(parents(5))  # ten prefixes, two dropped per epoch.
    batches = list(deterministic_batches(examples, batch_size=4, steps=7, seed=9))
    assert len(batches) == 7 and all(len(batch) == 4 for batch in batches)
    assert batches == list(deterministic_batches(examples, batch_size=4, steps=7, seed=9))
    assert batches != list(deterministic_batches(examples, batch_size=4, steps=7, seed=10))
    identity = lambda example: (example.parent_user_id, example.prefix_target_index)
    for start in (0, 2, 4):
        assert len({identity(e) for batch in batches[start:start + 2] for e in batch}) == 8
    for batch in batches:
        assert len({identity(e) for e in batch}) == 4
    assert sum(len(batch) for batch in batches) == 28


def test_batch_pool_rejects_too_small_duplicate_and_conflicting_parent():
    examples = expand_training_records(parents(2))
    with pytest.raises(ValueError, match='at least'):
        list(deterministic_batches(examples, batch_size=5))
    with pytest.raises(ValueError, match='Duplicate supervised'):
        list(deterministic_batches([examples[0]] * 4, batch_size=4))
    examples[1] = replace(examples[1], parent_serialized_sha256='f' * 64)
    with pytest.raises(ValueError, match='Conflicting'):
        list(deterministic_batches(examples, batch_size=4))


def test_generation_metrics_prefix_counts_natural_order_and_binary_duplicate_gt():
    labels = np.tile([1, 4097, 8193, 12289], (5, 1))
    tokens = np.zeros((5, 20, 4), dtype=np.int64)
    tokens[0, 0] = tokens[0, 2] = labels[0]  # Duplicate GT is one first-rank hit.
    tokens[1, 5] = labels[1]  # Rank six.
    tokens[2, 19] = labels[2]  # Rank twenty.
    tokens[3, 0, :3] = labels[3, :3]  # Three-level prefix, no GT hit.
    tokens[4, 0, :1] = labels[4, :1]  # One-level prefix only.
    result = generation_metrics(tokens, labels)
    assert result['prefix_hit_counts'] == [5, 4, 4, 3]
    assert result['duplicate_gt_queries'] == 1 and result['duplicate_policy'] == 'binary_first_hit'
    assert result['num_examples'] == 5
    assert result['Recall@5'] == 1 / 5 and result['Recall@10'] == 2 / 5
    assert result['Recall@20'] == 3 / 5
    np.testing.assert_allclose(result['NDCG@5'], 1 / 5)
    np.testing.assert_allclose(result['NDCG@10'], (1 + 1 / np.log2(7)) / 5)
    np.testing.assert_allclose(result['NDCG@20'], (1 + 1 / np.log2(7) + 1 / np.log2(21)) / 5)


def test_generation_metrics_zero_hits_are_zero_not_rank_zero():
    result = generation_metrics(np.zeros((2, 20, 4), dtype=np.int64), np.ones((2, 4), dtype=np.int64))
    assert result['prefix_hit_counts'] == [0, 0, 0, 0]
    assert all(result[key] == 0 for key in result if '@' in key)


@pytest.mark.parametrize('shape', [(0, 20, 4), (2, 19, 4), (2, 20, 3), (2, 80)])
def test_generation_metrics_rejects_wrong_shapes(shape):
    with pytest.raises(ValueError, match='Expected'):
        generation_metrics(np.zeros(shape, dtype=np.int64), np.ones((2, 4), dtype=np.int64))


def test_generation_metrics_rejects_float_ids_zero_gt_and_negative_tokens():
    tokens, labels = np.zeros((2, 20, 4), dtype=np.int64), np.ones((2, 4), dtype=np.int64)
    for invalid_tokens, invalid_labels in ((tokens.astype(float), labels), (tokens, labels.astype(float)),
                                           (tokens, labels * 0), (tokens - 1, labels)):
        with pytest.raises(ValueError, match='Expected'):
            generation_metrics(invalid_tokens, invalid_labels)


@pytest.mark.parametrize('bad', [True, 0, -1, 1.5])
def test_positive_budget_checks(bad):
    with pytest.raises(ValueError):
        split_parents(parents(2), fit_count=bad, reserved_count=1)
    with pytest.raises(ValueError):
        diagnostic_catalog(parents(1), np.arange(10), size=bad)
    with pytest.raises(ValueError):
        list(deterministic_batches(expand_training_records(parents(2)), batch_size=bad))
