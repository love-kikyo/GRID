"""Export dispatch and strict checkpoint/source contracts, without large models."""
from types import SimpleNamespace
import sys

import hydra
import pytest
import torch
from omegaconf import OmegaConf

from scripts import export_paired_evaluation as export


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.setup_calls = 0

    def setup(self, stage):
        assert stage == 'validate'
        self.teacher_model = torch.nn.Linear(1, 1)
        self.setup_calls += 1


def test_sim_strict_restore_constructs_teacher_before_load(tmp_path, monkeypatch):
    saved = ToyModel()
    saved.setup('validate')
    with torch.no_grad():
        saved.weight.fill_(3)
        saved.teacher_model.weight.fill_(7)
    path = tmp_path / 'sim.ckpt'
    torch.save({'state_dict': saved.state_dict()}, path)
    model = ToyModel()
    monkeypatch.setattr(hydra.utils, 'instantiate', lambda cfg: model)
    cfg = OmegaConf.create({'model': {'_target_': 'example.SIMRerankModule'}})
    actual = export.instantiate_model(cfg, path, torch.device('cpu'))
    assert actual.setup_calls == 1 and not actual.training
    for key, value in saved.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[key], value)
    broken = dict(saved.state_dict())
    broken.pop('weight')
    torch.save({'state_dict': broken}, path)
    monkeypatch.setattr(hydra.utils, 'instantiate', lambda cfg: ToyModel())
    with pytest.raises(RuntimeError, match='Missing key'):
        export.instantiate_model(cfg, path, torch.device('cpu'))


@pytest.mark.parametrize('name', ['SIMRerankModule', 'SeparateRerankModule'])
def test_frozen_reranker_uses_its_own_infer_entry(monkeypatch, name):
    sentinel = object()
    fake_mode = SimpleNamespace(INFER=sentinel)
    monkeypatch.setitem(sys.modules, 'src.models.modules.huggingface.transformer_base_module',
                        SimpleNamespace(ModelMode=fake_mode))
    model = type(name, (), {'model_step': lambda self, model_input, mode: (model_input, mode)})()
    batch = object()
    monkeypatch.setattr(export, 'move_batch_to_device', lambda value, device: value)
    assert export.infer_batch(model, batch, torch.device('cpu'), 'none') == (batch, sentinel)


def test_sim_source_contract_does_not_change_original_models():
    assert export.frozen_reranker_sources('example.SemanticIDEncoderDecoder') == []
    separate = export.frozen_reranker_sources('example.SeparateRerankModule')
    assert [p.name for p in separate] == ['separate_rerank_model.py']
    sim = export.frozen_reranker_sources('example.SIMRerankModule')
    assert [p.name for p in sim] == ['separate_rerank_model.py', 'sim_rerank_model.py',
                                   'sim_soft.py', 'tiger_train_sim_soft.yaml']
