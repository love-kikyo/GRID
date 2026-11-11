import copy

import numpy as np
import pytest
import torch

from scripts import retrieval_content_core as core
from test_retrieval_diagnostic_core import FakeModel, batch  # Reuse frozen test fixture.


def test_five_policies_one_search_and_independent_cache(batch):
    model = FakeModel()
    outputs = core.infer_content(model, batch, 'cpu')
    assert tuple(outputs) == core.POLICIES
    assert len(model.searches) == 1 and model.cache_inputs == [0] * 5
    for arrays in outputs.values():
        assert len(arrays) == 12
        np.testing.assert_array_equal(arrays['candidate_position'], [32, 32])
        np.testing.assert_array_equal(arrays['rerank_input_length'], [13, 13])
    assert (model.top_k_for_score, model.active_top_k_for_score) == (10, None)
    assert 'get_similar_itemkey' not in model.__dict__


def test_real_smoke_proves_frozen_subset_equivalence(batch):
    gate = core.smoke_gate(FakeModel(), batch, 'cpu')
    assert gate['status'] == 'passed' and gate['subset_matches_frozen_diagnostic'] is True
    assert gate['policies'] == list(core.POLICIES)
    assert set(gate['candidate_positions']) == set(core.POLICIES)
    assert set(gate['input_lengths']) == set(core.POLICIES)
    for key in ('native_sim10_exact', 'native_active_masks_exact', 'gt_independence_all_policies',
                'cache_order_independence', 'actual_rotary_positions_observed',
                'parameters_unchanged_and_gradients_absent'):
        assert gate[key] is True


@pytest.mark.parametrize('field', ['user_id', 'example_hash', 'target_sid', 'beam_sid', 'beam_score',
                                 'history_valid_count', 'candidate_position', 'rerank_input_length'])
def test_frozen_control_corruption_rejected(batch, field):
    outputs = core.infer_content(FakeModel(), batch, 'cpu')
    outputs['recent10'][field] = outputs['recent10'][field].copy()
    outputs['recent10'][field].flat[0] += 1
    with pytest.raises(AssertionError, match='changed'):
        core.check_controls(outputs)


@pytest.mark.parametrize('policy', ['mask0', 'mask5'])
def test_masked_content_corruption_rejected(batch, policy):
    outputs = core.infer_content(FakeModel(), batch, 'cpu')
    outputs[policy]['context_itemkey'] = outputs[policy]['context_itemkey'].copy()
    outputs[policy]['context_itemkey'][0, 0, 9] = 123
    with pytest.raises(AssertionError, match='Masked context changed'):
        core.check_controls(outputs)


def test_policy_order_and_membership_are_closed(batch):
    outputs = core.infer_content(FakeModel(), batch, 'cpu')
    with pytest.raises(AssertionError, match='policies or order'):
        core.check_controls(dict(reversed(list(outputs.items()))))
    outputs['short0'] = copy.deepcopy(outputs['mask0'])
    with pytest.raises(AssertionError, match='policies or order'):
        core.check_controls(outputs)


def test_smoke_detects_divergent_new_entry(monkeypatch, batch):
    original = core.infer_content
    def corrupt(*args, **kwargs):
        outputs = original(*args, **kwargs)
        outputs['random10']['raw_rerank_logit'][0, 0] += 1
        return outputs
    monkeypatch.setattr(core, 'infer_content', corrupt)
    with pytest.raises(AssertionError, match='Frozen diagnostic differs'):
        core.smoke_gate(FakeModel(), batch, 'cpu')


def test_smoke_covers_new_entry_parameter_mutation(monkeypatch, batch):
    original = core.infer_content
    def corrupt(model, *args, **kwargs):
        outputs = original(model, *args, **kwargs)
        with torch.no_grad():
            model.weight.add_(1.)
        return outputs
    monkeypatch.setattr(core, 'infer_content', corrupt)
    with pytest.raises(AssertionError, match='changed parameters or gradients'):
        core.smoke_gate(FakeModel(), batch, 'cpu')
