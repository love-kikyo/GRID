import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from src.baselines.liger_full_catalog import RowCatalog, full_catalog_cross_entropy


def fixture(dropout=0.):
    torch.manual_seed(71)
    model = nn.Sequential(nn.Linear(5, 9), nn.LayerNorm(9), nn.ReLU(),
                          nn.Dropout(dropout), nn.Linear(9, 4)).double()
    query = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
    content = torch.randn(11, 5, dtype=torch.double)
    return model, query, content, torch.tensor([0, 7, 10])


def reference(query, model, content, labels, chunk_size=100, masked_count=0):
    # Independent retained-graph broadcast implementation, not streamed logsumexp.
    projected = torch.cat([model(content[i:i + chunk_size]) for i in range(0, len(content), chunk_size)])
    logits = (F.normalize(query, dim=-1)[:, None] * F.normalize(projected, dim=-1)[None]).sum(-1) / .07
    if masked_count:
        logits = torch.cat([logits, logits.new_full((len(query), masked_count), -100)], dim=1)
    return F.cross_entropy(logits, labels)


@pytest.mark.parametrize('chunk_size', [1, 4, 11, 20])
@pytest.mark.parametrize('masked_count', [0, 6])
def test_full_loss_and_all_gradients_equal_monolithic_broadcast(chunk_size, masked_count):
    model, query, content, labels = fixture()
    a = reference(query, model, content, labels, masked_count=masked_count)
    ga = torch.autograd.grad(a, (query, *model.parameters()))
    b = full_catalog_cross_entropy(query, labels, model, content, chunk_size=chunk_size, masked_count=masked_count)
    gb = torch.autograd.grad(b, (query, *model.parameters()))
    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
    for x, y in zip(ga, gb):
        torch.testing.assert_close(x, y, rtol=1e-10, atol=1e-11)


def test_dropout_replay_scaled_shared_gradients_and_rng_restoration():
    a, _, content, labels = fixture(.2)
    b = copy.deepcopy(a)
    query_a = a(content[:3])  # The real LIGER projection is shared with input encoding.
    query_b = query_a.detach().clone().requires_grad_(True)
    rng = torch.get_rng_state()
    expected = reference(query_a, a, content, labels, chunk_size=4)
    torch.set_rng_state(rng)
    actual = full_catalog_cross_entropy(query_b, labels, b, content, chunk_size=4)
    after_forward = torch.get_rng_state()
    # Accumulate query-path + target-path gradients, with a nonunit upstream scale.
    qgrad, *bgrad = torch.autograd.grad(actual * 17, (query_b, *b.parameters()))
    assert torch.equal(torch.get_rng_state(), after_forward)
    direct = torch.autograd.grad(expected * 17, (query_a, *a.parameters()), retain_graph=True)
    torch.testing.assert_close(direct[0], qgrad, rtol=1e-10, atol=1e-10)
    input_grad = torch.autograd.grad(query_a, tuple(a.parameters()), grad_outputs=qgrad)
    for total, target, inputs in zip(direct[1:], bgrad, input_grad):
        torch.testing.assert_close(total, target + inputs, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(expected, actual, rtol=1e-12, atol=1e-12)


def test_shared_projection_accumulates_both_paths_in_one_backward():
    a, _, content, labels = fixture()
    b = copy.deepcopy(a)
    expected = reference(a(content[:3]), a, content, labels)
    actual = full_catalog_cross_entropy(b(content[:3]), labels, b, content, chunk_size=4)
    expected.backward()
    actual.backward()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-10, atol=1e-10)


def test_every_row_is_read_twice_and_largest_device_chunk_is_bounded():
    model, query, content, labels = fixture()
    class RecordingCatalog:
        calls = []
        def __len__(self):
            return len(content)
        def __getitem__(self, selection):
            self.calls.append((selection.start, selection.stop))
            return content[selection]
    catalog = RecordingCatalog()
    full_catalog_cross_entropy(query, labels, model, catalog, chunk_size=4).backward()
    assert catalog.calls == [(0, 4), (4, 8), (8, 11)] * 2


def test_row_catalog_preserves_zero_and_asset_identity():
    content = np.arange(50, dtype=np.int8).reshape(10, 5)
    rows = np.array([0, 2, 9], dtype=np.int32)
    catalog = RowCatalog(content, rows)
    assert len(catalog) == 3
    np.testing.assert_array_equal(catalog[:2], content[[0, 2]])
    for bad in [np.array([2, 2]), np.array([-1, 2]), np.array([0, 10]), np.array([0., 1.])]:
        with pytest.raises(ValueError):
            RowCatalog(content, bad)


@pytest.mark.parametrize('bad', ['label', 'chunk', 'masked', 'temperature', 'content_grad', 'buffers', 'dtype'])
def test_invalid_contract_rejected(bad):
    model, query, content, labels = fixture()
    kwargs = {}
    if bad == 'label': labels[0] = 11
    if bad == 'chunk': kwargs['chunk_size'] = 0
    if bad == 'masked': kwargs['masked_count'] = -1
    if bad == 'temperature': kwargs['temperature'] = float('nan')
    if bad == 'content_grad': content.requires_grad_(True)
    if bad == 'buffers': model.register_buffer('counter', torch.zeros(1))
    if bad == 'dtype': query = query.float()
    with pytest.raises(ValueError):
        full_catalog_cross_entropy(query, labels, model, content, **kwargs)


def test_mode_change_before_backward_rejected():
    model, query, content, labels = fixture(.2)
    loss = full_catalog_cross_entropy(query, labels, model, content)
    model.eval()
    with pytest.raises(RuntimeError, match='mode changed'):
        loss.backward()


def test_fp32_loss_and_gradients_match_reference():
    model, query, content, labels = fixture()
    model, query, content = model.float(), query.detach().float().requires_grad_(True), content.float()
    expected = reference(query, model, content, labels)
    actual = full_catalog_cross_entropy(query, labels, model, content, chunk_size=4)
    a = torch.autograd.grad(expected, (query, *model.parameters()))
    b = torch.autograd.grad(actual, (query, *model.parameters()))
    torch.testing.assert_close(expected, actual, atol=2e-6, rtol=2e-6)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=2e-5, rtol=2e-5)


def test_masked_constant_mass_is_preserved_not_dropped():
    model, query, content, labels = fixture()
    # Huge multiplicity makes the otherwise tiny exp(-100) mass observable.
    count = 10**45
    import math
    logits = (F.normalize(query, dim=-1)[:, None] * F.normalize(model(content), dim=-1)[None]).sum(-1) / .07
    reference_logits = torch.cat([logits, query.new_full((len(query), 1), math.log(count) - 100)], dim=1)
    expected = F.cross_entropy(reference_logits, labels)
    actual = full_catalog_cross_entropy(query, labels, model, content, chunk_size=4, masked_count=count)
    torch.testing.assert_close(expected, actual, rtol=1e-12, atol=1e-12)
    for x, y in zip(torch.autograd.grad(expected, (query, *model.parameters())),
                    torch.autograd.grad(actual, (query, *model.parameters()))):
        torch.testing.assert_close(x, y, rtol=1e-10, atol=1e-11)


def test_autocast_rejected_instead_of_silently_changing_dense_precision():
    model, query, content, labels = fixture()
    with torch.autocast('cpu', dtype=torch.bfloat16), pytest.raises(ValueError, match='outside autocast'):
        full_catalog_cross_entropy(query, labels, model, content)
