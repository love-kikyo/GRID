from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.baselines.liger_candidates import SidRowIndex
from src.baselines.liger_inference import generate_queries


class Model:
    training = False
    def __init__(self, early=False):
        self.early = early

    def generate(self, input_ids, **kwargs):
        assert 'labels' not in kwargs and kwargs['max_new_tokens'] == 4 and kwargs['num_beams'] == 20
        count = len(input_ids) * 20
        self.predicted_embedding = torch.ones(count, 3)
        item = [0, 16385] if self.early else [0, 1, 4097, 8193, 12289]
        return torch.tensor([item] * count)


def forward(model, batch, device, depth, method, skip_forward):
    assert skip_forward and depth == 4 and (batch['labels_sids'] == 0).all()
    return None, dict(input_ids=batch['input_sids'])


def inputs(gt):
    return dict(input_sids=np.ones((2, 29), np.int64), attention_mask_sids=np.ones((2, 29), np.int64),
                input_embeddings=np.ones((2, 7, 3), np.float32), labels_sids=np.full((2, 4), gt))


def test_generation_cannot_read_real_gt_and_sanitizes_duplicate_beams():
    index = SidRowIndex.build(np.zeros((1, 4), np.int32))
    eval_module = SimpleNamespace(model_forward=forward)
    query, a = generate_queries(Model(), inputs(777), {}, eval_module, index, 'cpu')
    other, b = generate_queries(Model(), inputs(123), {}, eval_module, index, 'cpu')
    assert torch.equal(query, other)
    for key in a: np.testing.assert_array_equal(a[key], b[key])
    assert a['raw_beam_rows'].tolist() == [[0] * 20] * 2
    assert a['valid_beam_rows'].tolist() == [[0] + [-1] * 19] * 2


def test_early_eos_is_invalid_not_row_zero():
    index = SidRowIndex.build(np.zeros((1, 4), np.int32))
    _, result = generate_queries(Model(early=True), inputs(5), {}, SimpleNamespace(model_forward=forward), index, 'cpu')
    assert (result['raw_beam_rows'] == -1).all()


def test_training_mode_rejected():
    model = Model()
    model.training = True
    with pytest.raises(ValueError):
        generate_queries(model, {}, {}, None, None, 'cpu')
