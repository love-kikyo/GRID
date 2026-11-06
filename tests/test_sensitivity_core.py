"""CPU fake-model gates for fixed-checkpoint sensitivity inference."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import sensitivity_core as core


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.top_k_for_generation = 20
        self.top_k_for_score = 10
        self.active_top_k_for_score = None
        self.num_hierarchies = 4
        self.num_embeddings_per_hierarchy = 64
        self.powers = torch.tensor([64**3, 64**2, 64, 1])
        self.beam_log_prob_weight = 1.0
        self.rerank_logit_weight = 0.1
        self.searches, self.histories, self.cache_inputs = [], [], []
        self.fail_m = None
        self.eval()

    def _beam_search_semantic_ids(self, *, sid, attention_mask):
        c, b = self.top_k_for_generation, len(sid)
        self.searches.append((c, sid.clone(), attention_mask.clone()))
        # Width is part of candidate identity: C10 must not be a C20 slice.
        generated = torch.ones((b, c, 4), dtype=torch.long)
        generated[:, :, 0] = c
        generated[:, :, -1] = torch.arange(c) + 1
        scores = -torch.arange(c).float().expand(b, -1)
        return generated.reshape(-1, 4), scores, torch.ones((b*c, 3)), {"count": 0}

    def clone_kv_cache(self, cache):
        return copy.deepcopy(cache)

    def _rerank_with_click_prediction(self, *, generated_ids, beam_log_prob, input_mask,
                                      past_key_values, hist_itemkey, batch_size):
        self.cache_inputs.append(past_key_values["count"])
        self.histories.append(hist_itemkey.clone())
        past_key_values["count"] += 1  # deliberately mutate each supplied cache
        if self.top_k_for_score == self.fail_m:
            raise RuntimeError("injected failure")
        c = self.top_k_for_generation
        logits = torch.full((batch_size, c), float(self.top_k_for_score + past_key_values["count"]))
        scores = beam_log_prob + 0.1*logits
        return generated_ids.reshape(batch_size, c, 4), scores, logits

    def _infer_for_eval(self, *, hist_sid, hist_mask, hist_itemkey, target_sid):
        ids, scores, mask, cache = self._beam_search_semantic_ids(sid=hist_sid, attention_mask=hist_mask)
        rids, rscores, logits = self._rerank_with_click_prediction(
            generated_ids=ids, beam_log_prob=scores, input_mask=mask, past_key_values=cache,
            hist_itemkey=hist_itemkey, batch_size=len(hist_sid))
        return {"beam": {"ids": ids.reshape(len(hist_sid), self.top_k_for_generation, 4), "scores": scores},
                "rerank": {"ids": rids, "scores": rscores, "click_logits": logits}}, torch.tensor(0.)


@pytest.fixture
def batch():
    return SimpleNamespace(user_id_list=torch.tensor([11, 12]), mask=torch.ones((2, 12), dtype=torch.long),
        transformed_sequences={"sequence_data": torch.arange(24).reshape(2, 12)+1,
                               "hist_itemkey": torch.tensor([[91, 92, 999], [81, 82, 888]])})


def test_retrieval_shares_one_search_and_clones_cache(batch):
    model = FakeModel()
    result = core.infer_settings(model, batch, "cpu", "retrieval")
    assert len(model.searches) == 1
    assert model.cache_inputs == [0, 0, 0, 0]
    assert list(result) == ["c20_m0", "c20_m5", "c20_m10", "c20_m20"]
    for name, arrays in result.items():
        assert arrays["beam_sid"].shape == (2, 20, 4)
        assert arrays["raw_rerank_logit"].dtype == np.float32
        np.testing.assert_array_equal(arrays["beam_sid"], result["c20_m10"]["beam_sid"])
        np.testing.assert_array_equal(arrays["beam_score"], result["c20_m10"]["beam_score"])
    assert model.top_k_for_score == 10 and model.top_k_for_generation == 20
    assert model.active_top_k_for_score is None


def test_each_beam_width_runs_natural_search(batch):
    model = FakeModel()
    result = core.infer_settings(model, batch, "cpu", "beam")
    assert [r[0] for r in model.searches] == [10, 40]
    assert result["c10_m10"]["beam_sid"].shape == (2, 10, 4)
    assert result["c40_m10"]["beam_sid"].shape == (2, 40, 4)
    assert not np.array_equal(result["c10_m10"]["beam_sid"], result["c40_m10"]["beam_sid"][:, :10])


def test_gt_is_excluded_from_every_model_input(batch):
    model = FakeModel()
    core.infer_settings(model, batch, "cpu", "retrieval")
    torch.testing.assert_close(model.searches[0][1], batch.transformed_sequences["sequence_data"][:, :-4])
    for history in model.histories:
        torch.testing.assert_close(history, batch.transformed_sequences["hist_itemkey"][:, :-1])


def test_restore_on_failure(batch):
    model = FakeModel()
    model.active_top_k_for_score = 5
    model.fail_m = 20
    with pytest.raises(RuntimeError, match="injected"):
        core.infer_settings(model, batch, "cpu", "retrieval")
    assert (model.top_k_for_generation, model.top_k_for_score, model.active_top_k_for_score) == (20, 10, 5)


def test_smoke_checks_original_path_and_label_independence(batch):
    model = FakeModel()
    receipt = core.smoke_gate(model, batch, "cpu")
    assert receipt["status"] == "passed"
    assert receipt["gt_independence_all_settings"]
    assert len(receipt["all_settings"]) == 6
    assert model.weight.grad is None
    assert all(x == 0 for x in model.cache_inputs)


def test_smoke_rejects_cache_alias(batch):
    model = FakeModel()
    model.clone_kv_cache = lambda cache: cache
    with pytest.raises(AssertionError, match="differs"):
        core.smoke_gate(model, batch, "cpu")


def test_bad_group_and_training_mode_rejected(batch):
    model = FakeModel()
    with pytest.raises(ValueError, match="Unknown"):
        core.infer_settings(model, batch, "cpu", "other")
    model.train()
    with pytest.raises(ValueError, match="eval"):
        core.infer_settings(model, batch, "cpu", "retrieval")
