"""Small CPU tensor tests; run on the compute node due to Lightning/TF imports."""
from types import SimpleNamespace

import pytest
import torch

from scripts.verify_cpu_resume import CPUResumeAudit, assert_equal


def test_tensor_optimizer_equality_rejects_reset():
    expected = {'state': {0: {'step': torch.tensor(20000.), 'exp_avg': torch.tensor([.1])}}}
    assert_equal(expected, expected)
    reset = {'state': {0: {'step': torch.tensor(0.), 'exp_avg': torch.tensor([.1])}}}
    with pytest.raises(AssertionError, match='tensor mismatch'):
        assert_equal(reset, expected)


@pytest.mark.parametrize('actual,expected', [({'a': 1}, {'b': 1}), ([1], [1, 2]), (0, 20000)])
def test_structural_or_scalar_mismatch_is_fatal(actual, expected):
    with pytest.raises(AssertionError): assert_equal(actual, expected)


def test_smoke_requires_exact_final_step(tmp_path):
    audit = CPUResumeAudit('unused.ckpt', tmp_path, 20000, 100)
    trainer = SimpleNamespace(global_step=100)
    with pytest.raises(AssertionError, match='intended resumed'):
        audit.on_fit_end(trainer, None)


def test_short_smoke_cannot_pass_without_real_batches(tmp_path):
    audit = CPUResumeAudit('unused.ckpt', tmp_path, 20000, 100)
    trainer = SimpleNamespace(global_step=20100)
    with pytest.raises(AssertionError, match='actual training'):
        audit.on_fit_end(trainer, None)
