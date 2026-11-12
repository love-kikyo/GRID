"""CPU mathematical checks of one candidate draw per logical optimizer step."""
import copy
import gc
import weakref

import numpy as np
import pytest
import torch
from torch import nn

from src.baselines.liger_logical_batch import prepare_candidate_pool, microbatch_dense_loss
from src.baselines.liger_sampled_catalog import sampled_catalog_cross_entropy


def fixture(size=7, dropout=0.):
    torch.manual_seed(318)
    projection = nn.Sequential(nn.Linear(5, 9), nn.LayerNorm(9), nn.ReLU(),
                               nn.Dropout(dropout), nn.Linear(9, 4)).double()
    content = torch.randn(29, 5, dtype=torch.float64)
    queries = torch.randn(size, 5, dtype=torch.float64)
    labels = (torch.arange(size) * 7 % 29).long()
    if size > 2: labels[-1] = labels[0]
    return projection, content, queries, labels


@pytest.mark.parametrize('size,micro', [(7, 1), (7, 3), (7, 7), (81, 32), (81, 81), (1, 512)])
@pytest.mark.parametrize('negatives', [4, 100])
@pytest.mark.parametrize('masked', [0, 10**45])
def test_logical_loss_and_shared_query_projection_gradients_match_monolithic(size, micro, negatives, masked):
    projection, content, query_features, labels = fixture(size)
    reference = copy.deepcopy(projection)
    expected_query = reference(query_features)
    expected = sampled_catalog_cross_entropy(expected_query, labels, reference, content,
        num_negatives=negatives, seed=91, attempt=13, temperature=.19, masked_count=masked)
    expected.backward()
    pool = prepare_candidate_pool(projection, content, labels, num_negatives=negatives,
        seed=91, attempt=13, temperature=.19, masked_count=masked)
    total = 0.
    for begin in range(0, size, micro):
        end = min(size, begin + micro)
        query = projection(query_features[begin:end])
        loss = microbatch_dense_loss(query, pool, begin, end)
        total += loss.item()
        loss.backward(retain_graph=end < size)
        del query, loss
    assert total == pytest.approx(expected.item(), abs=1e-12)
    for a, b in zip(projection.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-10, atol=1e-10)


def test_dropout_candidate_draw_is_once_shared_and_micro_loss_does_not_advance_rng():
    projection, content, _, labels = fixture(dropout=.4)
    query = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    reference = copy.deepcopy(projection)
    calls = []
    hook = projection.register_forward_hook(lambda *args: calls.append(1))
    before = torch.get_rng_state()
    pool = prepare_candidate_pool(projection, content, labels, num_negatives=4, seed=3)
    after = torch.get_rng_state()
    loss_sum = 0.
    for start, stop in ((0, 3), (3, 7)):
        loss = microbatch_dense_loss(query[start:stop], pool, start, stop)
        loss_sum += loss.item()
        loss.backward(retain_graph=stop < 7)
    assert calls == [1] and torch.equal(torch.get_rng_state(), after)
    torch.set_rng_state(before)
    other_query = query.detach().clone().requires_grad_()
    expected = sampled_catalog_cross_entropy(other_query, labels, reference, content,
                                             num_negatives=4, seed=3)
    expected.backward()
    assert torch.equal(torch.get_rng_state(), after)
    assert loss_sum == pytest.approx(expected.item(), abs=1e-12)
    torch.testing.assert_close(query.grad, other_query.grad, rtol=1e-10, atol=1e-10)
    for a, b in zip(projection.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-10, atol=1e-10)
    hook.remove()


def test_retained_candidate_graph_does_not_keep_released_query_graph_alive():
    projection, content, features, labels = fixture()
    pool = prepare_candidate_pool(projection, content, labels, num_negatives=4, seed=3)
    query = projection(features[:3])
    observed = weakref.ref(query)
    loss = microbatch_dense_loss(query, pool, 0, 3)
    loss.backward(retain_graph=True)
    del query, loss
    gc.collect()
    assert observed() is None
    microbatch_dense_loss(projection(features[3:]), pool, 3, 7).backward()


def test_combined_sid_mean_weighting_and_one_adam_update_match_full_logical_batch():
    projection, content, features, labels = fixture(9)
    sid_head = nn.Linear(4, 11).double()
    projection_ref, sid_ref = copy.deepcopy(projection), copy.deepcopy(sid_head)
    opt = torch.optim.AdamW([*projection.parameters(), *sid_head.parameters()], lr=3e-4)
    opt_ref = torch.optim.AdamW([*projection_ref.parameters(), *sid_ref.parameters()], lr=3e-4)
    targets = torch.arange(9) % 11
    for attempt in range(2):
        opt.zero_grad(set_to_none=True); opt_ref.zero_grad(set_to_none=True)
        full_query = projection_ref(features)
        expected = (torch.nn.functional.cross_entropy(sid_ref(full_query), targets)
            + sampled_catalog_cross_entropy(full_query, labels, projection_ref, content,
                                             num_negatives=5, seed=8, attempt=attempt))
        expected.backward()
        pool = prepare_candidate_pool(projection, content, labels, num_negatives=5,
                                      seed=8, attempt=attempt)
        for start, stop in ((0, 4), (4, 8), (8, 9)):
            query = projection(features[start:stop])
            sid = torch.nn.functional.cross_entropy(sid_head(query), targets[start:stop])
            loss = (sid * pool.microbatch_weight(start, stop)
                    + microbatch_dense_loss(query, pool, start, stop))
            loss.backward(retain_graph=stop < len(labels))
            del query, sid, loss
        torch.nn.utils.clip_grad_norm_([*projection.parameters(), *sid_head.parameters()], 1.)
        torch.nn.utils.clip_grad_norm_([*projection_ref.parameters(), *sid_ref.parameters()], 1.)
        opt.step(); opt_ref.step()
        for actual, reference in zip([*projection.parameters(), *sid_head.parameters()],
                                     [*projection_ref.parameters(), *sid_ref.parameters()]):
            torch.testing.assert_close(actual, reference, rtol=1e-11, atol=1e-12)
        assert {int(state['step']) for state in opt.state.values()} == {attempt + 1}


def test_forced_positive_union_contains_other_microbatch_targets_and_duplicates_share_class():
    projection, content, _, _ = fixture()
    labels = torch.tensor([0, 28, 11, 0, 7, 28])
    pool = prepare_candidate_pool(projection, content, labels, num_negatives=3, seed=8)
    np.testing.assert_array_equal(pool.sample.positions[pool.sample.target_indices], labels.numpy())
    assert pool.sample.forced_positives == 4 and pool.sample.sampled_negatives == 3
    assert pool.sample.target_indices[0] == pool.sample.target_indices[3]
    assert pool.sample.target_indices[1] == pool.sample.target_indices[5]
    for target in labels.unique():
        assert int(target) in pool.sample.positions
    assert pool.logical_size == 6
    assert pool.microbatch_weight(0, 4) == 4 / 6
    assert pool.microbatch_weight(4, 6) == 2 / 6


def test_tail_81_and_nonuniform_microbatch_weighting_include_all_examples_once():
    projection, content, _, labels = fixture(81)
    pool = prepare_candidate_pool(projection, content, labels, num_negatives=3, seed=8)
    assert sum(pool.microbatch_weight(a, b) for a, b in ((0, 32), (32, 64), (64, 81))) == 1
    assert pool.logical_size == 81 and pool.microbatch_weight(0, 81) == 1


def test_frozen_catalog_access_is_bounded_and_once_per_logical_batch():
    projection, _, _, labels = fixture()
    class VirtualCatalog:
        calls = []
        def __len__(self): return 10**12
        def __getitem__(self, selection):
            self.calls.append(np.array(selection))
            return np.cos(np.asarray(selection)[:, None] % 97 + np.arange(5)[None])
    catalog = VirtualCatalog()
    pool = prepare_candidate_pool(projection, catalog, labels, num_negatives=11, seed=32)
    query = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    for begin, end in ((0, 4), (4, 7)):
        microbatch_dense_loss(query[begin:end], pool, begin, end).backward(retain_graph=end < 7)
    assert len(catalog.calls) == 1
    assert len(catalog.calls[0]) == len(labels.unique()) + 11


@pytest.mark.parametrize('mutation', ['parameter', 'mode', 'replace_parameter', 'dtype', 'requires_grad'])
def test_pool_rejects_projection_changes_before_next_microbatch(mutation):
    projection, content, _, labels = fixture()
    pool = prepare_candidate_pool(projection, content, labels, num_negatives=4, seed=3)
    if mutation == 'parameter':
        with torch.no_grad(): projection[0].weight.add_(.01)
    elif mutation == 'mode': projection.eval()
    elif mutation == 'replace_parameter': projection[0].weight = nn.Parameter(projection[0].weight.clone())
    elif mutation == 'dtype': projection.float()
    elif mutation == 'requires_grad': projection[0].weight.requires_grad_(False)
    with pytest.raises(ValueError, match='cannot cross'):
        microbatch_dense_loss(torch.ones(7, 4, dtype=torch.float64), pool, 0, 7)


@pytest.mark.parametrize('start,stop', [(-1, 3), (0, 0), (4, 3), (0, 8), (False, 3), (0, 3.5)])
def test_invalid_logical_slices(start, stop):
    projection, content, _, labels = fixture()
    pool = prepare_candidate_pool(projection, content, labels, num_negatives=4, seed=3)
    with pytest.raises(ValueError, match='nonempty slice'):
        pool.microbatch_weight(start, stop)


@pytest.mark.parametrize('kind', ['shape', 'dtype', 'nonfinite'])
def test_invalid_queries(kind):
    projection, content, _, labels = fixture()
    pool = prepare_candidate_pool(projection, content, labels, num_negatives=4, seed=3)
    query = torch.ones(3, 4, dtype=torch.float64)
    if kind == 'shape': query = query[:, :3]
    elif kind == 'dtype': query = query.float()
    else: query[0, 0] = float('nan')
    with pytest.raises(ValueError, match='Query must match'):
        microbatch_dense_loss(query, pool, 0, 3)


@pytest.mark.parametrize('kwargs', [{'temperature': 0}, {'temperature': True},
    {'temperature': float('inf')}, {'masked_count': -1}, {'masked_count': False},
    {'num_negatives': 0}, {'seed': -1}, {'attempt': -1}])
def test_invalid_sampling_controls(kwargs):
    projection, content, _, labels = fixture()
    controls = dict(num_negatives=4, seed=3); controls.update(kwargs)
    with pytest.raises(ValueError): prepare_candidate_pool(projection, content, labels, **controls)


@pytest.mark.parametrize('kind', ['labels_dtype', 'empty_labels', 'labels_out_of_range',
    'catalog_grad', 'raw_nonfinite', 'projection_frozen', 'projection_buffer'])
def test_invalid_preparation(kind):
    projection, content, _, labels = fixture()
    if kind == 'labels_dtype': labels = labels.float()
    elif kind == 'empty_labels': labels = labels[:0]
    elif kind == 'labels_out_of_range': labels[0] = len(content)
    elif kind == 'catalog_grad': content.requires_grad_()
    elif kind == 'raw_nonfinite': content[0, 0] = float('nan')
    elif kind == 'projection_frozen': projection.requires_grad_(False)
    elif kind == 'projection_buffer': projection.register_buffer('counter', torch.tensor(0))
    with pytest.raises(ValueError): prepare_candidate_pool(projection, content, labels, num_negatives=4, seed=3)


def test_grad_disabled_and_autocast_are_rejected():
    projection, content, _, labels = fixture()
    with torch.no_grad(), pytest.raises(ValueError, match='gradients enabled'):
        prepare_candidate_pool(projection, content, labels, num_negatives=4, seed=3)
    projection.float(); content = content.float()
    with torch.autocast('cpu', dtype=torch.bfloat16), pytest.raises(ValueError, match='outside autocast'):
        prepare_candidate_pool(projection, content, labels, num_negatives=4, seed=3)
