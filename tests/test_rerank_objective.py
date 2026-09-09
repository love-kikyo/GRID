from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torchmetrics.aggregation import MeanMetric

from src.models.modules.huggingface.transformer_base_module import ModelMode
from src.models.modules.semantic_id.separate_rerank_model import SeparateRerankModule
from src.models.modules.semantic_id.tiger_generation_model import (
    SemanticIDEncoderDecoder,
)


def _minimal_model(objective):
    model = SemanticIDEncoderDecoder.__new__(SemanticIDEncoderDecoder)
    nn.Module.__init__(model)
    model.rerank_objective = objective
    model.click_loss_fn = nn.BCEWithLogitsLoss()
    model.use_beam_score_in_rerank_loss = False
    return model


def test_pointwise_and_listwise_share_valid_candidate_rows():
    logits = torch.tensor([[2.0, 0.0], [5.0, -5.0]], requires_grad=True)
    labels = torch.tensor([[1.0, 0.0], [0.0, 0.0]])

    pointwise = _minimal_model("pointwise")._compute_rerank_training_loss(
        logits, labels
    )
    listwise = _minimal_model("listwise")._compute_rerank_training_loss(
        logits, labels
    )

    expected_pointwise = nn.BCEWithLogitsLoss()(logits[:1], labels[:1])
    expected_listwise = -torch.log_softmax(logits[:1], dim=-1)[0, 0]
    torch.testing.assert_close(pointwise, expected_pointwise)
    torch.testing.assert_close(listwise, expected_listwise)


def test_objectives_return_differentiable_zero_without_positive_list():
    logits = torch.tensor([[1.0, -1.0]], requires_grad=True)
    labels = torch.zeros_like(logits)

    for objective in ("pointwise", "listwise"):
        loss = _minimal_model(objective)._compute_rerank_training_loss(logits, labels)
        assert loss.item() == 0.0
        loss.backward(retain_graph=True)
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


class _AuxTeacherModel(nn.Module):
    def _compute_target_sid_log_prob(self, hist_sid, hist_mask, target_sid):
        return torch.full(
            (target_sid.size(0),),
            -5.0,
            device=target_sid.device,
        )


class _AuxTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.sid_model = _AuxTeacherModel()


class _RecordingAuxReranker(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_candidate_itemkey = None

    def forward(self, hist_itemkey, candidate_itemkey, candidate_sid, beam_scores):
        self.seen_candidate_itemkey = candidate_itemkey.detach().clone()
        return torch.zeros_like(beam_scores, requires_grad=True)


def test_separate_reranker_matches_training_only_no_hit_aux_rule():
    model = SeparateRerankModule.__new__(SeparateRerankModule)
    nn.Module.__init__(model)
    model.top_k_for_generation = 2
    model.use_beam_score_in_rerank_loss = False
    model.use_no_hit_aux_loss = True
    model.no_hit_aux_loss_weight = 0.5
    model.teacher = _AuxTeacher()
    model.reranker = _RecordingAuxReranker()
    model._safe_binary_auroc = lambda probabilities, labels: probabilities.sum() * 0.0

    batch_dict = {
        "hist_sid": torch.ones(2, 4, dtype=torch.long),
        "hist_mask": torch.ones(2, 4, dtype=torch.long),
        "hist_itemkey_ctx": torch.tensor([[1, 2], [3, 4]]),
        "target_sid": torch.tensor([[1, 1, 1, 1], [9, 9, 9, 9]]),
        "target_itemkey": torch.tensor([11, 99]),
    }
    natural_candidate_itemkey = torch.tensor([[11, 12], [21, 22]])
    natural_result = {
        "candidate_sid": torch.ones(2, 2, 4, dtype=torch.long),
        "candidate_itemkey": natural_candidate_itemkey.clone(),
        "beam_scores": torch.tensor([[0.0, -1.0], [0.0, -1.0]]),
        "labels": torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
        "rerank_logits": torch.tensor(
            [[2.0, 0.0], [0.0, 0.0]],
            requires_grad=True,
        ),
        "final_scores": torch.tensor([[2.0, 0.0], [0.0, 0.0]]),
    }
    model._split_batch = lambda model_input: batch_dict

    def _natural_beam_only(**kwargs):
        assert kwargs["ensure_gt_in_candidates"] is False
        return natural_result

    model._run_rerank = _natural_beam_only
    torch.manual_seed(7)

    _, loss, metrics = model.model_step(object(), ModelMode.TRAIN)

    expected_main = -torch.log_softmax(
        natural_result["rerank_logits"][:1], dim=-1
    )[0, 0]
    expected_aux = torch.log(torch.tensor(2.0))
    torch.testing.assert_close(loss, expected_main + 0.5 * expected_aux)
    torch.testing.assert_close(metrics["rerank_aux_loss"], expected_aux)
    assert metrics["rerank_no_hit_rate"].item() == 0.5
    # The auxiliary scorer sees the injected target, but the saved natural
    # candidates remain unchanged for inference and paired evaluation.
    assert 99 in model.reranker.seen_candidate_itemkey[0].tolist()
    torch.testing.assert_close(
        natural_result["candidate_itemkey"],
        natural_candidate_itemkey,
    )


class _RecordingEvaluator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)


def _separate_eval_fixture():
    model = SeparateRerankModule.__new__(SeparateRerankModule)
    nn.Module.__init__(model)
    model.num_hierarchies = 2
    model.powers = torch.tensor([16, 1])
    model.val_loss = MeanMetric()
    model.click_log_prob_accumulator = MeanMetric()
    model.beam_log_prob_accumulator = MeanMetric()
    model.click_auc_accumulator = MeanMetric()
    model.evaluators = {name: _RecordingEvaluator() for name in ("beam", "rerank")}
    batch = SimpleNamespace(transformed_sequences={
        "sequence_data": torch.tensor([[9, 9, 1, 2], [8, 8, 7, 7]])
    })
    ids = torch.tensor([[[1, 2], [2, 3], [3, 4]], [[1, 1], [2, 2], [3, 3]]])
    result = {
        "beam": {"ids": ids, "scores": torch.tensor([[-1., -2., -4.], [-3., -4., -5.]])},
        "rerank": {
            "ids": ids.flip(1),
            "scores": torch.tensor([[1000., 999., 997.], [-2., -4., -5.]]),
            "click_logits": torch.tensor([[0., 1., 2.], [-1., -2., -3.]]),
        },
    }
    modes = []

    def model_step(model_input, mode):
        assert model_input is batch
        modes.append(mode)
        if mode is ModelMode.TRAIN:
            return None, torch.tensor(0.75), {}
        assert mode is ModelMode.INFER
        return result

    model.model_step = model_step
    return model, batch, result, modes


@pytest.mark.parametrize("wrapped", [False, True])
def test_separate_eval_updates_log_prob_metrics_without_changing_predictions(wrapped):
    model, batch, result, modes = _separate_eval_fixture()
    original = deepcopy(result)
    model.on_validation_epoch_start()
    for _ in range(2):
        model.eval_step((batch, None) if wrapped else batch, model.val_loss)

    assert modes == [ModelMode.TRAIN, ModelMode.INFER] * 2
    torch.testing.assert_close(model.val_loss.compute(), torch.tensor(0.75))
    torch.testing.assert_close(
        model.beam_log_prob_accumulator.compute(), result["beam"]["scores"].mean()
    )
    # Match joint logging: normalized fused scores, not raw click logits.
    torch.testing.assert_close(
        model.click_log_prob_accumulator.compute(),
        torch.log_softmax(result["rerank"]["scores"], dim=-1).mean(),
    )
    assert torch.isfinite(model.click_auc_accumulator.compute())
    for name, evaluator in model.evaluators.items():
        assert len(evaluator.calls) == 2
        for call in evaluator.calls:
            torch.testing.assert_close(call["marginal_probs"], original[name]["scores"])
            torch.testing.assert_close(call["generated_ids"], original[name]["ids"])
        for key, value in result[name].items():
            torch.testing.assert_close(value, original[name][key])


def test_separate_validation_log_prob_metrics_reset_between_epochs():
    model, batch, result, _ = _separate_eval_fixture()
    model.on_validation_epoch_start()
    model.eval_step(batch, model.val_loss)
    model.on_validation_epoch_start()
    result["beam"]["scores"] = torch.full((2, 3), -7.)
    result["rerank"]["scores"] = torch.zeros(2, 3)
    model.eval_step(batch, model.val_loss)
    torch.testing.assert_close(model.beam_log_prob_accumulator.compute(), torch.tensor(-7.))
    torch.testing.assert_close(
        model.click_log_prob_accumulator.compute(), -torch.log(torch.tensor(3.))
    )
