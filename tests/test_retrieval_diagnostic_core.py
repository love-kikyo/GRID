import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import retrieval_diagnostic_core as core


class Rotary(torch.nn.Module):
    def forward(self, x, position_ids):
        return x


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Module()
        self.decoder.rotary_emb = Rotary()
        self.compress_positions = False

    def forward(self, *, sequence_embedding, attention_mask, past_key_values=None, use_cache=False):
        length = sequence_embedding.shape[1]
        position = torch.arange(20, 20 + length)[None, :]
        if self.compress_positions:
            position = 19 + attention_mask[:, -length:].long().cumsum(-1)
        self.decoder.rotary_emb(sequence_embedding, position)
        return sequence_embedding


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))
        self.decoder = Decoder()
        self.top_k_for_generation, self.top_k_for_score, self.active_top_k_for_score = 20, 10, None
        self.num_hierarchies, self.num_embeddings_per_hierarchy, self.masking_token = 4, 64, -1
        self.powers = torch.tensor([64**3, 64**2, 64, 1])
        self.beam_log_prob_weight, self.rerank_logit_weight = 1., .1
        self.searches, self.cache_inputs, self.histories = [], [], []
        self.fail_slots = None
        self.eval()

    def _beam_search_semantic_ids(self, *, sid, attention_mask):
        self.searches.append(sid.clone())
        b, c = len(sid), 20
        ids = torch.ones((b, c, 4), dtype=torch.long)
        ids[:, :, -1] = torch.arange(1, c + 1)
        return ids.reshape(-1, 4), -torch.arange(c).float().expand(b, -1), torch.ones((b*c, 3)), {'count': 0}

    def clone_kv_cache(self, cache):
        return copy.deepcopy(cache)

    def get_similar_itemkey(self, target_itemkey, hist_itemkey):
        k = self.top_k_for_score
        if k == 0:
            return target_itemkey[:, None]
        scores = -(hist_itemkey - (target_itemkey % 200)[:, None]).abs().float()
        scores.masked_fill_(hist_itemkey == -1, -1e9)
        chosen = hist_itemkey.gather(1, torch.topk(scores, k, dim=-1).indices)
        if self.active_top_k_for_score is not None:
            chosen[:, self.active_top_k_for_score:] = -1
        return torch.cat([chosen, target_itemkey[:, None]], -1)

    def _rerank_with_click_prediction(self, *, generated_ids, beam_log_prob, input_mask,
                                      past_key_values, hist_itemkey, batch_size):
        self.cache_inputs.append(past_key_values['count'])
        past_key_values['count'] += 1
        self.histories.append(hist_itemkey.clone())
        if self.top_k_for_score == self.fail_slots:
            raise RuntimeError('injected rerank failure')
        target = (generated_ids * self.powers).sum(-1)
        context = self.get_similar_itemkey(target_itemkey=target, hist_itemkey=hist_itemkey.repeat_interleave(20, 0))
        embeds = torch.cat([generated_ids[:, -1:], target[:, None], context], -1).float().unsqueeze(-1)
        attention = torch.cat([input_mask, torch.ones((len(target), 1)), context.ne(-1)], -1)
        self.decoder(sequence_embedding=embeds, attention_mask=attention, past_key_values=past_key_values)
        raw = ((context[:, :-1].clamp_min(0).sum(-1).float() / 100
                + (target % 13).float() / 2 + past_key_values['count'] * 10
                + embeds.shape[1]) / 10).reshape(batch_size, 20)
        fused = beam_log_prob + .1 * raw
        score, order = fused.topk(20, dim=-1)
        ids = generated_ids.reshape(batch_size, 20, 4).gather(1, order[:, :, None].expand(-1, -1, 4))
        return ids, score, raw.gather(1, order)

    def _infer_for_eval(self, *, hist_sid, hist_mask, hist_itemkey, target_sid):
        ids, scores, mask, cache = self._beam_search_semantic_ids(sid=hist_sid, attention_mask=hist_mask)
        rids, rscores, raw = self._rerank_with_click_prediction(generated_ids=ids, beam_log_prob=scores,
            input_mask=mask, past_key_values=cache, hist_itemkey=hist_itemkey, batch_size=len(hist_sid))
        return {'beam': {'ids': ids.reshape(len(hist_sid), 20, 4), 'scores': scores},
                'rerank': {'ids': rids, 'scores': rscores, 'click_logits': raw}}, torch.tensor(0.)


@pytest.fixture
def batch():
    return SimpleNamespace(user_id_list=torch.tensor([11, 12]), mask=torch.ones((2, 12), dtype=torch.long),
        transformed_sequences={'sequence_data': torch.arange(24).reshape(2, 12) + 1,
            'hist_itemkey': torch.tensor([list(range(101, 124)) + [101, 112, 9999],
                                          [-1]*22 + [211, 212, 213, 8888]])})


def test_shared_beam_positions_and_matched_content(batch):
    model = FakeModel()
    out = core.infer_diagnostics(model, batch, 'cpu')
    assert list(out) == list(core.POLICIES)
    assert len(model.searches) == 1 and model.cache_inputs == [0]*7
    for name, arrays in out.items():
        length = 3 if name == 'short0' else 8 if name == 'short5' else 13
        np.testing.assert_array_equal(arrays['rerank_input_length'], [length]*2)
        np.testing.assert_array_equal(arrays['candidate_position'], [19 + length]*2)
        np.testing.assert_array_equal(arrays['history_valid_count'], [25, 3])
        assert arrays['context_itemkey'].shape == (2, 20, 10)
        assert arrays['raw_rerank_logit'].dtype == np.float32
    for active in (0, 5):
        np.testing.assert_array_equal(out[f'mask{active}']['context_itemkey'], out[f'short{active}']['context_itemkey'])
    assert 'get_similar_itemkey' not in model.__dict__
    assert (model.top_k_for_generation, model.top_k_for_score, model.active_top_k_for_score) == (20, 10, None)


def test_real_input_hooks_and_complete_smoke(batch):
    model = FakeModel()
    gate = core.smoke_gate(model, batch, 'cpu')
    assert gate['status'] == 'passed' and gate['native_active_masks_exact']
    assert gate['actual_rotary_positions_observed'] and gate['gt_independence_all_policies']
    assert gate['input_lengths']['mask0'] == [13]
    assert not model.decoder._forward_pre_hooks
    assert not model.decoder.decoder.rotary_emb._forward_pre_hooks
    assert all(count == 0 for count in model.cache_inputs)


@pytest.mark.parametrize('policy', ['recent10', 'random10'])
def test_history_selection_is_user_stable_and_retains_duplicates(policy):
    histories = np.array([[10]*4 + list(range(20, 36)), [-1]*17 + [4, 4, 5]])
    first = core.select_history(histories, [90, 91], policy)
    flipped = core.select_history(histories[::-1], [91, 90], policy)
    np.testing.assert_array_equal(first, flipped[::-1])
    np.testing.assert_array_equal(first[1], [4, 4, 5] + [-1]*7)
    assert (first[0] != -1).sum() == 10
    assert set(first[0]) <= set(histories[0])
    if policy == 'recent10':
        np.testing.assert_array_equal(first[0], np.arange(26, 36))


def test_random_selection_does_not_use_global_rng():
    hist = np.arange(120).reshape(3, 40)
    one = core.select_history(hist, [1, 2, 3], 'random10')
    np.random.seed(999)
    np.random.random(1000)
    np.testing.assert_array_equal(one, core.select_history(hist, [1, 2, 3], 'random10'))
    assert not np.array_equal(one[0], core.select_history(hist[:1], [999], 'random10')[0])


@pytest.mark.parametrize('policy', ['recent10', 'random10'])
def test_no_valid_history_remains_padding(policy):
    np.testing.assert_array_equal(core.select_history(np.full((1, 25), -1), [123], policy),
                                  np.full((1, 10), -1))


def test_random_candidate_sets_shared_and_recent_is_latest(batch):
    model = FakeModel()
    out = core.infer_diagnostics(model, batch, 'cpu')
    for name in ('recent10', 'random10'):
        context = np.sort(out[name]['context_itemkey'], -1)
        np.testing.assert_array_equal(context, np.broadcast_to(context[:, :1], context.shape))
    expected = np.sort(batch.transformed_sequences['hist_itemkey'][0, -11:-1].numpy())
    np.testing.assert_array_equal(np.sort(out['recent10']['context_itemkey'][0, 0]), expected)
    for h in model.histories:
        assert not bool(((h == 9999) | (h == 8888)).any())


def test_failure_restores_methods_settings_and_hooks(batch):
    model = FakeModel()
    model.active_top_k_for_score = 7
    model.fail_slots = 5
    with pytest.raises(RuntimeError, match='injected'):
        core.infer_diagnostics(model, batch, 'cpu')
    assert model.top_k_for_score == 10 and model.active_top_k_for_score == 7
    assert 'get_similar_itemkey' not in model.__dict__
    assert not model.decoder._forward_pre_hooks and not model.decoder.decoder.rotary_emb._forward_pre_hooks


def test_original_instance_override_is_restored(batch):
    model = FakeModel()
    original = model.get_similar_itemkey
    model.get_similar_itemkey = original
    core.infer_diagnostics(model, batch, 'cpu')
    assert model.__dict__['get_similar_itemkey'] is original


def test_unavailable_or_unobserved_rotary_is_rejected(batch):
    model = FakeModel()
    del model.decoder.decoder.rotary_emb
    with pytest.raises(RuntimeError, match='position hook unavailable'):
        core.infer_diagnostics(model, batch, 'cpu')
    assert 'get_similar_itemkey' not in model.__dict__


def test_mask_compressed_position_ids_are_rejected(batch):
    model = FakeModel()
    model.decoder.compress_positions = True
    with pytest.raises(AssertionError, match='positions violate'):
        core.infer_diagnostics(model, batch, 'cpu')


def test_cache_alias_is_rejected(batch):
    model = FakeModel()
    model.clone_kv_cache = lambda cache: cache
    with pytest.raises(AssertionError, match='differs'):
        core.smoke_gate(model, batch, 'cpu')


def test_foreign_history_item_is_rejected(batch):
    model = FakeModel()
    original = model.get_similar_itemkey
    def corrupt(*args, **kwargs):
        value = original(*args, **kwargs)
        value[:, 0] = 9999999
        return value
    model.get_similar_itemkey = corrupt
    with pytest.raises(AssertionError, match='outside eligible history'):
        core.infer_diagnostics(model, batch, 'cpu')


def test_invalid_policy_or_training_rejected(batch):
    model = FakeModel()
    with pytest.raises(ValueError, match='Unsupported'):
        core._infer_sequence(model, batch, 'cpu', ['bad'])
    model.train()
    with pytest.raises(ValueError, match='eval'):
        core.infer_diagnostics(model, batch, 'cpu')
