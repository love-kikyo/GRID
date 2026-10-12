"""Run on compute nodes: the real model imports TensorFlow through data types."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.baselines.sim_soft import SIMSoftScorer
from src.models.modules.huggingface.transformer_base_module import ModelMode
from src.models.modules.semantic_id.sim_rerank_model import SIMRerankModule
from src.models.modules.semantic_id.separate_rerank_model import SeparateRerankModule


class TinyEmbedding(nn.Embedding):
    def __init__(self):
        super().__init__(64, 8)

    def forward(self, keys):
        return super().forward(keys.clamp_min(0))


class FakeTeacher(nn.Module):
    def generate_candidates(self, *args, **kwargs):
        assert kwargs['ensure_gt_in_candidates'] is False
        keys = torch.tensor([[4, 5], [4, 5]])
        target = args[4]
        return dict(candidate_itemkey=keys, candidate_sid=keys.unsqueeze(-1),
                    beam_scores=torch.tensor([[-1., -2.], [-1., -2.]]),
                    labels=(keys == target[:, None]).float())


def adapter():
    # Only bypass the large backbone allocation; exercise real adapter methods.
    m = SIMRerankModule.__new__(SIMRerankModule)
    nn.Module.__init__(m)
    m.reranker = SIMSoftScorer(TinyEmbedding(), 8, 3, 2)
    m.teacher = FakeTeacher()
    m.teacher_model = None
    m.top_k_for_generation, m.num_hierarchies = 2, 1
    m.beam_log_prob_weight, m.rerank_logit_weight = 1., .1
    m.use_no_hit_aux_loss, m.no_hit_aux_loss_weight, m.sim_gsu_loss_weight = True, .5, 1.
    return m


def batch(targets=(4, 9)):
    return SimpleNamespace(transformed_sequences={
        'sequence_data': torch.tensor([[1, 2, targets[0]], [7, 8, targets[1]]]),
        'hist_itemkey': torch.tensor([[1, 2, targets[0]], [7, 8, targets[1]]]),
    }, mask=torch.ones(2, 3, dtype=torch.bool))


def test_train_mode_never_enables_teacher_training():
    model = adapter().train()
    assert model.reranker.training and not model.teacher.training


def test_setup_freezes_new_teacher_wrapper_even_without_later_train_call(monkeypatch):
    model = adapter()
    def create_teacher(self, stage):
        self.teacher = FakeTeacher().train()
    monkeypatch.setattr(SeparateRerankModule, 'setup', create_teacher)
    model.setup('fit')
    assert not model.teacher.training


def test_targets_never_change_inference_candidates_or_scores():
    model = adapter().eval()
    a = model.model_step(batch(), ModelMode.INFER)
    b = model.model_step(batch((25, 26)), ModelMode.INFER)
    for method in a:
        for key in a[method]:
            torch.testing.assert_close(a[method][key], b[method][key], rtol=0, atol=0)


def test_loss_only_gt_injection_is_copy_and_has_no_beam_dependency():
    model = adapter()
    original = model.model_step(batch(), ModelMode.INFER)
    torch.manual_seed(12)
    _, loss, metrics = model.model_step(batch(), ModelMode.TRAIN)
    loss.backward()
    assert metrics['rerank_no_hit_rate'].item() == .5
    assert metrics['sim_gsu_loss'] > 0 and metrics['sim_esu_loss'] > 0
    for name in ('gsu_query', 'gsu_history', 'esu_query', 'esu_history'):
        assert torch.count_nonzero(getattr(model.reranker, name).weight.grad)
    again = model.model_step(batch(), ModelMode.INFER)
    torch.testing.assert_close(original['beam']['ids'], again['beam']['ids'])
    model.rerank_logit_weight = 99
    model.beam_log_prob_weight = -99
    torch.manual_seed(12)
    _, other_loss, _ = model.model_step(batch(), ModelMode.TRAIN)
    torch.testing.assert_close(loss, other_loss, rtol=0, atol=0)


def test_all_no_hit_still_trains_gsu_and_esu():
    model = adapter()
    _, loss, _ = model.model_step(batch((25, 26)), ModelMode.TRAIN)
    loss.backward()
    assert torch.isfinite(loss) and loss > 0
    assert torch.count_nonzero(model.reranker.gsu_query.weight.grad)
    assert torch.count_nonzero(model.reranker.esu_query.weight.grad)


def test_candidate_path_rejects_gt_injection():
    model = adapter()
    with pytest.raises(ValueError, match='forbids'):
        model._run_rerank(**model._split_batch(batch()), ensure_gt_in_candidates=True)
