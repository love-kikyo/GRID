import copy

import pytest
import torch
from torch import nn

from scripts.probe_liger_cpu4 import ROOT, import_official
from src.baselines.liger_projection import FullWidthLinear, use_linear_residuals


@pytest.mark.parametrize('batch', [1, 7])
@pytest.mark.parametrize('bias', [False, True])
def test_full_width_convolution_output_input_and_parameter_gradients(batch, bias):
    torch.manual_seed(91)
    conv = nn.Conv1d(1, 13, 5, stride=5, bias=bias).double()
    matrix = FullWidthLinear(conv)
    x = torch.randn(batch, 1, 5, dtype=torch.double, requires_grad=True)
    a, b = conv(x), matrix(x)
    assert matrix.weight is conv.weight and matrix.bias is conv.bias
    assert a.shape == b.shape == (batch, 13, 1)
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    for ga, gb in zip(torch.autograd.grad(a.square().mean(), (x, *conv.parameters())),
                      torch.autograd.grad(b.square().mean(), (x, *matrix.parameters()))):
        torch.testing.assert_close(ga, gb, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('training', [False, True])
def test_official_mlp_state_identity_dropout_and_all_gradients(training):
    official = import_official('_grid_test_liger_projection', ROOT.parent / 'liger-upstream/src/tiger.py')
    torch.manual_seed(73)
    a = official.MLP(128, [768, 512, 256], 128, dropout=.2).double().train(training)
    b = copy.deepcopy(a)
    parameters_before = list(b.parameters())
    assert use_linear_residuals(b) == 3
    assert all(p is q for p, q in zip(parameters_before, b.parameters()))
    assert list(a.state_dict()) == list(b.state_dict())
    b.load_state_dict(a.state_dict(), strict=True)
    x = torch.randn(7, 128, dtype=torch.double, requires_grad=True)
    rng = torch.get_rng_state()
    expected = a(x)
    after_a = torch.get_rng_state()
    torch.set_rng_state(rng)
    actual = b(x)
    assert torch.equal(after_a, torch.get_rng_state())
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
    for ga, gb in zip(torch.autograd.grad(expected.square().mean(), (x, *a.parameters())),
                      torch.autograd.grad(actual.square().mean(), (x, *b.parameters()))):
        torch.testing.assert_close(ga, gb, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize('conv', [nn.Conv1d(2, 3, 5, stride=5), nn.Conv1d(1, 3, 5, stride=1),
                                  nn.Conv1d(1, 3, 5, stride=5, padding=1)])
def test_other_convolution_geometries_are_rejected(conv):
    with pytest.raises(ValueError):
        FullWidthLinear(conv)


def test_wrong_input_width_is_rejected_instead_of_changing_semantics():
    matrix = FullWidthLinear(nn.Conv1d(1, 3, 5, stride=5))
    with pytest.raises(ValueError):
        matrix(torch.ones(2, 1, 10))


def test_invalid_later_residual_does_not_partly_mutate_model():
    model = nn.Module()
    model.residuals = nn.ModuleList([nn.Conv1d(1, 3, 5, stride=5), nn.Linear(3, 3)])
    with pytest.raises(ValueError):
        use_linear_residuals(model)
    assert isinstance(model.residuals[0], nn.Conv1d)
