import numpy as np
import pytest

from src.baselines.liger_inputs import (
    catalog_memory_lower_bounds, cosine_catalog_logits, pack_sequences,
)


def fixture():
    sid = np.arange(48).reshape(12, 4) % 8
    content = np.arange(60).reshape(12, 5).astype(np.int8)
    return sid, content


def test_last_item_is_target_and_history_is_not_expanded():
    sid, content = fixture()
    batch = pack_sequences([list(range(12)), [2, 3]], sid, content, codebook_size=8)
    assert batch['history_rows'] == [list(range(4, 11)), [2]]
    assert batch['target_rows'].tolist() == [11, 3]
    np.testing.assert_array_equal(batch['labels_sids'], sid[[11, 3]] + [1, 9, 17, 25])
    assert batch['input_sids'].shape == (2, 29)
    assert batch['attention_mask_sids'].sum(axis=1).tolist() == [28, 4]
    assert (batch['input_sids'][1, 4:] == 0).all()
    assert (batch['input_embeddings'][1, 1:] == 0).all()
    np.testing.assert_array_equal(batch['input_embeddings'][0], content[4:11])


def test_repeated_item_is_not_deduplicated_or_removed_from_history():
    sid, content = fixture()
    batch = pack_sequences([[1, 3, 1]], sid, content, codebook_size=8)
    assert batch['history_rows'] == [[1, 3]]
    assert batch['target_rows'].tolist() == [1]


@pytest.mark.parametrize('seq', [[], [1], [1, -1], [1, 12]])
def test_invalid_sequence_rejected(seq):
    with pytest.raises(ValueError):
        pack_sequences([seq], *fixture(), codebook_size=8)


def test_bad_assets_or_codebook_rejected():
    sid, content = fixture()
    with pytest.raises(ValueError, match='align'):
        pack_sequences([[0, 1]], sid, content[:-1])
    with pytest.raises(ValueError, match='codebook'):
        pack_sequences([[0, 1]], sid, content, codebook_size=4)


@pytest.mark.parametrize('zero', [False, True])
def test_dense_matmul_matches_broadcast_logits_loss_and_both_gradients(zero):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(42)
    query = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
    items = torch.randn(11, 8, dtype=torch.float64, requires_grad=True)
    if zero:
        with torch.no_grad():
            items[0].zero_()
    labels = torch.tensor([1, 2, 7])
    broadcast = (F.normalize(query, dim=1)[:, None, :]
                 * F.normalize(items, dim=-1)).sum(-1) / .07
    matrix = cosine_catalog_logits(query, items, .07)
    torch.testing.assert_close(matrix, broadcast, atol=1e-12, rtol=1e-12)
    a, b = F.cross_entropy(broadcast, labels), F.cross_entropy(matrix, labels)
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    for ga, gb in zip(torch.autograd.grad(a, (query, items)),
                      torch.autograd.grad(b, (query, items))):
        torch.testing.assert_close(ga, gb, atol=1e-3 if zero else 1e-12, rtol=1e-12)


def test_asset_row_memory_estimate_does_not_claim_active_catalog_size():
    bounds = catalog_memory_lower_bounds(35458467)
    assert bounds['broadcast_product_fp16_bytes'] > 250 * 1024**3
    assert bounds['projection_first_hidden_fp32_bytes'] > 100 * 1024**3
    with pytest.raises(ValueError):
        catalog_memory_lower_bounds(0)
