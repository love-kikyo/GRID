"""CPU-only mathematical and bounded-access checks for sampled dense training."""
import copy
from collections import Counter
from itertools import combinations, product
import math
import random

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from src.baselines.liger_full_catalog import RowCatalog, full_catalog_cross_entropy
from src.baselines.liger_sampled_catalog import (
    sample_candidates, sampled_catalog_cross_entropy,
)


def fixture(dtype=torch.float64, dropout=0.):
    torch.manual_seed(71)
    projection = nn.Sequential(nn.Linear(5, 9), nn.LayerNorm(9), nn.ReLU(),
                               nn.Dropout(dropout), nn.Linear(9, 4)).to(dtype)
    query = torch.randn(4, 4, dtype=dtype, requires_grad=True)
    content = torch.randn(13, 5, dtype=dtype)
    labels = torch.tensor([0, 7, 7, 12])
    return projection, query, content, labels


def independent_reference(query, projection, content, labels, positions, *, temperature=.07,
                          masked_count=0):
    """Broadcast logits and explicit inclusion probabilities, not production CE."""
    vectors = projection(content[positions])
    logits = (F.normalize(query, dim=-1)[:, None]
              * F.normalize(vectors, dim=-1)[None]).sum(-1) / temperature
    positives = set(labels.tolist())
    remaining = len(content) - len(positives)
    sampled = len(positions) - len(positives)
    probabilities = [1. if int(row) in positives else sampled / remaining for row in positions]
    corrected = logits - torch.tensor(probabilities, dtype=query.dtype).log()[None]
    if masked_count:
        # One synthetic class exactly represents the sum of masked exp(-100) logits.
        corrected = torch.cat([corrected, query.new_full((len(query), 1),
                              math.log(masked_count) - 100)], dim=1)
    targets = torch.tensor([list(positions).index(int(label)) for label in labels])
    return F.cross_entropy(corrected, targets)


@pytest.mark.parametrize('negatives', [10, 13, 100])
@pytest.mark.parametrize('masked_count', [0, 6, 10**45])
def test_full_sampling_loss_and_all_gradients_match_full_catalog(negatives, masked_count):
    projection, query, content, labels = fixture()
    expected = full_catalog_cross_entropy(query, labels, projection, content,
                                         chunk_size=4, masked_count=masked_count)
    actual = sampled_catalog_cross_entropy(query, labels, projection, content,
        num_negatives=negatives, seed=41, masked_count=masked_count)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    for a, b in zip(torch.autograd.grad(actual, (query, *projection.parameters())),
                    torch.autograd.grad(expected, (query, *projection.parameters()))):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('masked_count', [0, 10**45])
def test_partial_sampling_matches_independent_corrected_loss_and_gradients(dtype, masked_count):
    projection, query, content, labels = fixture(dtype)
    selected = sample_candidates(labels, len(content), 3, seed=42, attempt=9)
    expected = independent_reference(query, projection, content, labels, selected.positions,
                                     temperature=.19, masked_count=masked_count)
    actual = sampled_catalog_cross_entropy(query, labels, projection, content,
        num_negatives=3, seed=42, attempt=9, temperature=.19, masked_count=masked_count)
    tolerance = 3e-6 if dtype == torch.float32 else 1e-11
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    for a, b in zip(torch.autograd.grad(actual * 11., (query, *projection.parameters())),
                    torch.autograd.grad(expected * 11., (query, *projection.parameters()))):
        torch.testing.assert_close(a, b, atol=tolerance * 10, rtol=tolerance * 10)


def test_forced_positives_are_unique_and_not_false_negatives():
    labels = torch.tensor([9, 0, 9, 4, 0])
    sample = sample_candidates(labels, 17, 5, seed=3)
    assert sample.population_size == 17 and sample.forced_positives == 3
    assert sample.remaining_population == 14 and sample.sampled_negatives == 5
    assert sample.positions.dtype == np.int64 and sample.target_indices.dtype == np.int64
    assert sample.log_inclusion.dtype == np.float64
    assert len(sample.positions) == 8 and np.all(np.diff(sample.positions) > 0)
    np.testing.assert_array_equal(sample.positions[sample.target_indices], labels.numpy())
    positive_mask = np.isin(sample.positions, labels.numpy())
    np.testing.assert_array_equal(sample.log_inclusion[positive_mask], np.zeros(3))
    np.testing.assert_allclose(sample.log_inclusion[~positive_mask], math.log(5 / 14), atol=0, rtol=0)
    assert sample.target_indices[0] == sample.target_indices[2]
    assert sample.target_indices[1] == sample.target_indices[4]


@pytest.mark.parametrize('labels,size,negatives', [([0], 1, 1), ([2, 0, 1, 2], 3, 1),
                                                  ([0, 2], 3, 100)])
def test_empty_or_exhausted_complement_is_exact(labels, size, negatives):
    sample = sample_candidates(torch.tensor(labels), size, negatives, seed=0)
    np.testing.assert_array_equal(sample.positions, np.arange(size))
    np.testing.assert_array_equal(sample.log_inclusion, np.zeros(size))
    np.testing.assert_array_equal(sample.positions[sample.target_indices], labels)
    assert sample.sampled_negatives == sample.remaining_population


def test_seed_and_attempt_are_reproducible_and_do_not_consume_global_rng():
    labels = torch.tensor([0, 200, 999])
    python_before, numpy_before, torch_before = random.getstate(), np.random.get_state(), torch.get_rng_state()
    first = sample_candidates(labels, 1000, 31, seed=17, attempt=4)
    repeat = sample_candidates(labels, 1000, 31, seed=17, attempt=4)
    next_attempt = sample_candidates(labels, 1000, 31, seed=17, attempt=5)
    next_seed = sample_candidates(labels, 1000, 31, seed=18, attempt=4)
    np.testing.assert_array_equal(first.positions, repeat.positions)
    assert not np.array_equal(first.positions, next_attempt.positions)
    assert not np.array_equal(first.positions, next_seed.positions)
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0] and numpy_after[2:] == numpy_before[2:]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_label_permutation_or_duplication_does_not_change_shared_negatives():
    a = sample_candidates(torch.tensor([2, 5, 9]), 100, 11, seed=4, attempt=3)
    b = sample_candidates(torch.tensor([9, 5, 2, 9]), 100, 11, seed=4, attempt=3)
    np.testing.assert_array_equal(a.positions, b.positions)
    np.testing.assert_array_equal(a.log_inclusion, b.log_inclusion)


def test_wide_seed_does_not_alias_the_attempt_entropy_word():
    labels = torch.tensor([0, 9])
    wide_seed = sample_candidates(labels, 10000, 31, seed=2**32, attempt=0)
    shifted_attempt = sample_candidates(labels, 10000, 31, seed=0, attempt=1)
    repeat = sample_candidates(labels, 10000, 31, seed=2**32, attempt=0)
    assert not np.array_equal(wide_seed.positions, shifted_attempt.positions)
    np.testing.assert_array_equal(wide_seed.positions, repeat.positions)


def test_exhaustive_floyd_paths_are_uniform_and_partition_not_log_loss_is_unbiased(monkeypatch):
    # R=4, K=2: Floyd makes independent uniform draws in [0,2] and [0,3].
    # All twelve equiprobable paths can be checked, without statistical tolerance.
    from src.baselines import liger_sampled_catalog as implementation
    labels = torch.tensor([1, 3, 1])
    weights = np.array([.5, 3., 1., 5., 4., 7.])
    counts, partitions = Counter(), []
    for draws in product(range(3), range(4)):
        class FixedDraws:
            def __init__(self): self.draws = iter(draws)
            def integers(self, low, high):
                value = next(self.draws)
                assert low == 0 and low <= value < high
                return value
        monkeypatch.setattr(implementation.np.random, 'Generator', lambda _: FixedDraws())
        sample = sample_candidates(labels, 6, 2, seed=0)
        negatives = tuple(int(i) for i in sample.positions if i not in {1, 3})
        counts[negatives] += 1
        partitions.append(np.sum(weights[sample.positions] * np.exp(-sample.log_inclusion)))
    assert counts == Counter({pair: 2 for pair in combinations([0, 2, 4, 5], 2)})
    assert np.mean(partitions) == pytest.approx(weights.sum(), abs=1e-14)
    assert np.mean(np.log(partitions)) < np.log(weights.sum())


@pytest.mark.parametrize('positives', [[0, 1, 4, 5, 7], [0, 1, 2, 3, 4], [3, 4, 5, 6, 7]])
def test_exhaustive_complement_rank_mapping_handles_adjacent_positive_runs(monkeypatch, positives):
    from src.baselines import liger_sampled_catalog as implementation
    observed = []
    for rank in range(3):
        class FixedDraw:
            def integers(self, low, high):
                assert (low, high) == (0, 3)
                return rank
        monkeypatch.setattr(implementation.np.random, 'Generator', lambda _: FixedDraw())
        sample = sample_candidates(torch.tensor(positives), 8, 1, seed=0)
        observed.extend(int(i) for i in sample.positions if i not in positives)
    assert observed == [i for i in range(8) if i not in positives]


def test_huge_virtual_catalog_only_reads_sampled_rows_once():
    projection, query, _, _ = fixture()
    count = 10**12
    labels = torch.tensor([0, 7, 7, count - 1])
    class VirtualCatalog:
        calls = []
        def __len__(self): return count
        def __getitem__(self, selection):
            assert not isinstance(selection, slice), 'Cannot scan a huge catalog'
            indices = np.asarray(selection)
            assert len(indices) <= 14 and np.all((indices >= 0) & (indices < count))
            self.calls.append(indices.copy())
            return np.cos(indices[:, None] % 97 + np.arange(5)[None]).astype('float64')
    catalog = VirtualCatalog()
    loss = sampled_catalog_cross_entropy(query, labels, projection, catalog,
                                         num_negatives=11, seed=32)
    loss.backward()
    assert len(catalog.calls) == 1 and len(catalog.calls[0]) == 14
    assert torch.isfinite(loss) and torch.isfinite(query.grad).all()


def test_catalog_positions_are_not_global_asset_rows():
    projection, query, content, _ = fixture()
    seen = np.array([0, 2, 5, 8, 12])
    catalog = RowCatalog(content.numpy(), seen)
    labels = torch.tensor([0, 3, 3, 4])
    selected = sample_candidates(labels, len(seen), 1, seed=9)
    expected = independent_reference(query, projection, content[seen], labels, selected.positions)
    actual = sampled_catalog_cross_entropy(query, labels, projection, catalog,
                                         num_negatives=1, seed=9)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('dropout', [0., .25])
def test_shared_projection_accumulates_query_and_item_gradients_with_dropout(dropout):
    projection, _, content, labels = fixture(dropout=dropout)
    reference = copy.deepcopy(projection)
    rng = torch.get_rng_state()
    query = projection(content[:4])
    actual = sampled_catalog_cross_entropy(query, labels, projection, content,
                                         num_negatives=4, seed=3)
    after_forward = torch.get_rng_state()
    actual.backward()
    assert torch.equal(after_forward, torch.get_rng_state())
    torch.set_rng_state(rng)
    expected_query = reference(content[:4])
    sample = sample_candidates(labels, len(content), 4, seed=3)
    expected = independent_reference(expected_query, reference, content, labels, sample.positions)
    expected.backward()
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    for a, b in zip(projection.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('key,value', [('catalog_size', 0), ('catalog_size', True),
    ('catalog_size', 2.5), ('num_negatives', 0), ('num_negatives', -1),
    ('num_negatives', True), ('num_negatives', np.bool_(True)), ('num_negatives', 1.5),
    ('seed', -1), ('seed', True), ('seed', 1.5), ('attempt', -1), ('attempt', True),
    ('attempt', 1.5), ('labels', torch.tensor([], dtype=torch.long)),
    ('labels', torch.tensor([[-1]])), ('labels', torch.tensor([-1])),
    ('labels', torch.tensor([13])), ('labels', torch.tensor([1.], dtype=torch.float32)),
    ('labels', torch.tensor([1], dtype=torch.int32))])
def test_sampler_rejects_invalid_contract(key, value):
    args = dict(labels=torch.tensor([0, 7, 12]), catalog_size=13, num_negatives=4, seed=7, attempt=0)
    args[key] = value
    with pytest.raises(ValueError): sample_candidates(**args)


@pytest.mark.parametrize('bad', ['empty_batch', 'batch_shape', 'half_query', 'query_nan',
    'content_nan', 'content_grad', 'projection_nan', 'buffers', 'dtype', 'temperature_zero',
    'temperature_nan', 'temperature_infinite', 'masked_negative', 'masked_bool'])
def test_dense_loss_rejects_invalid_contract(bad):
    projection, query, content, labels = fixture()
    kwargs = dict(num_negatives=3, seed=4)
    if bad == 'empty_batch': query, labels = query[:0], labels[:0]
    if bad == 'batch_shape': labels = labels[:2]
    if bad == 'half_query': query = query.half()
    if bad == 'query_nan': query = query.detach().clone(); query[0, 0] = float('nan')
    if bad == 'content_nan': content[:] = float('nan')
    if bad == 'content_grad': content.requires_grad_(True)
    if bad == 'projection_nan':
        with torch.no_grad(): next(projection.parameters()).fill_(float('nan'))
    if bad == 'buffers': projection.register_buffer('counter', torch.zeros(1))
    if bad == 'dtype': projection.float()
    if bad == 'temperature_zero': kwargs['temperature'] = 0.
    if bad == 'temperature_nan': kwargs['temperature'] = float('nan')
    if bad == 'temperature_infinite': kwargs['temperature'] = float('inf')
    if bad == 'masked_negative': kwargs['masked_count'] = -1
    if bad == 'masked_bool': kwargs['masked_count'] = True
    with pytest.raises(ValueError):
        sampled_catalog_cross_entropy(query, labels, projection, content, **kwargs)


def test_autocast_is_rejected_not_silently_used_for_dense_scores():
    projection, query, content, labels = fixture(torch.float32)
    with torch.autocast('cpu', dtype=torch.bfloat16), pytest.raises(ValueError):
        sampled_catalog_cross_entropy(query, labels, projection, content, num_negatives=3, seed=1)
