import pytest
import torch
from torch import nn

from scripts.verify_stage_smoke import compare_teacher_parameters, check_gradient_groups


def test_teacher_comparison_requires_all_exact_parameters():
    model = nn.Linear(2, 2)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    assert compare_teacher_parameters(model, state) == 6
    state['weight'][0, 0] += 1
    with pytest.raises(AssertionError, match='not loaded exactly'):
        compare_teacher_parameters(model, state)


@pytest.mark.parametrize('method', ['listwise', 'pointwise', 'separate'])
def test_gradient_audit_checks_frozen_parameters_and_active_groups(method):
    model = nn.Module()
    model.decoder = nn.Module()
    model.decoder.sid_head = nn.Linear(2, 1)
    model.decoder.click_head = nn.Linear(2, 1)
    model.backbone = nn.Linear(2, 2)
    if method == 'separate':
        model.reranker = nn.Linear(2, 1)
    for name, param in model.named_parameters():
        param.requires_grad = method != 'separate' or name.startswith('reranker.')
        if param.requires_grad:
            param.grad = torch.ones_like(param)
    assert all(check_gradient_groups(model, method).values())
    parameter = next(p for p in model.parameters() if p.requires_grad)
    parameter.grad.fill_(float('nan'))
    with pytest.raises(AssertionError, match='Nonfinite'):
        check_gradient_groups(model, method)
