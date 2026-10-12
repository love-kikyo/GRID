import pytest
import torch
from torch import nn

from src.baselines.sim_soft import SIMSoftScorer, supervised_bce


class TinyEmbedding(nn.Embedding):
    def __init__(self, dim=8):
        super().__init__(64, dim)

    def forward(self, keys):
        return super().forward(keys.clamp_min(0))


def scorer(top_k=3):
    torch.manual_seed(42)
    return SIMSoftScorer(TinyEmbedding(), 8, top_k, 2)


def test_retrieval_is_learned_dot_product_and_masks_padding():
    model = scorer(2)
    hist, cand = torch.tensor([[-1, 1, 2, 3]]), torch.tensor([[4, 5]])
    result = model.score(hist, cand, auxiliary=True)
    scores = model.gsu_query(model.item_embedding(cand)) @ model.gsu_history(model.item_embedding(hist)).transpose(1, 2)
    scores[:, :, 0] = float('-inf')
    assert torch.equal(result['selected_indices'], scores.topk(2).indices)
    assert result['selected_mask'].all()


@pytest.mark.parametrize('history', [[-1, -1, -1], [1], []])
def test_short_empty_and_all_padding_histories_are_finite(history):
    model = scorer(10)
    result = model.score(torch.tensor([history], dtype=torch.long), torch.tensor([[4, 5]]), auxiliary=True)
    loss = result['esu_logits'].sum() + result['gsu_logits'].sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert result['selected_mask'].sum().item() == (2 if history == [1] else 0)


def test_gsu_pooling_is_unnormalized_paper_equation_two():
    model = scorer()
    hist, cand = torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5]])
    result = model.score(hist, cand, auxiliary=True)
    h, q = model.item_embedding(hist), model.item_embedding(cand)
    r = model.gsu_query(q) @ model.gsu_history(h).transpose(1, 2)
    expected = model.gsu_head(torch.cat((r @ h, q), -1)).squeeze(-1)
    torch.testing.assert_close(result['gsu_logits'], expected)


def test_padding_invariance_and_candidate_permutation_equivariance():
    model = scorer()
    hist, cand = torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6]])
    plain = model(hist, cand)
    torch.testing.assert_close(plain, model(torch.tensor([[-1, -1, 1, 2, 3]]), cand))
    torch.testing.assert_close(plain[:, [2, 0, 1]], model(hist, cand[:, [2, 0, 1]]))


def test_no_sid_rank_or_beam_feature_and_no_input_mutation():
    model = scorer()
    hist, cand = torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5]])
    before_h, before_c = hist.clone(), cand.clone()
    plain = model(hist, cand)
    torch.testing.assert_close(plain, model(hist, cand, torch.randn(1, 2, 4), torch.randn(1, 2)))
    assert torch.equal(before_h, hist) and torch.equal(before_c, cand)


def test_auxiliary_loss_trains_both_search_projections_and_heads():
    model = scorer()
    result = model.score(torch.tensor([[1, 2, 3], [7, 8, 9]]), torch.tensor([[4, 5], [10, 11]]), auxiliary=True)
    labels = torch.tensor([[1., 0.], [0., 1.]])
    loss = sum(supervised_bce(result[key], labels) for key in ('esu_logits', 'gsu_logits'))
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert torch.count_nonzero(p.grad), name


def test_no_hit_bce_is_excluded_and_graph_remains_differentiable():
    logits = torch.tensor([[1., -1.], [2., -2.]], requires_grad=True)
    labels = torch.tensor([[1., 0.], [0., 0.]])
    torch.testing.assert_close(supervised_bce(logits, labels),
                               torch.nn.functional.binary_cross_entropy_with_logits(logits[:1], labels[:1]))
    loss = supervised_bce(logits, torch.zeros_like(labels))
    loss.backward()
    assert loss.item() == 0 and torch.count_nonzero(logits.grad) == 0


def test_invalid_dimensions_fail_closed():
    with pytest.raises(ValueError):
        scorer(0)
