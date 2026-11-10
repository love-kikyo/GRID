"""Small tensor tests: run on the allocated compute node (Lightning imports TF)."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.verify_random_init import (
    GROUPS, PHASES, RandomInitAudit, assert_fresh, parameter_fingerprints,
    update_evidence, verify_reports,
)


def model():
    module = nn.Module()
    module.backbone = nn.Linear(2, 2)
    module.decoder = nn.Module()
    module.decoder.sid_head = nn.Linear(2, 1)
    module.decoder.click_head = nn.Linear(2, 1)
    module.load_only_weights = False
    module._manual_ckpt_path = None
    module.training_task = 'click'
    module.rerank_objective = 'listwise'
    return module


def trainer(module):
    return SimpleNamespace(world_size=3, global_rank=0, global_step=0, current_epoch=0,
                           ckpt_path=None, optimizers=[torch.optim.AdamW(module.parameters(), lr=1e-3)])


def test_full_hash_detects_change_and_does_not_mutate_parameters():
    module = model()
    initial = parameter_fingerprints(module)
    assert parameter_fingerprints(module) == initial
    with torch.no_grad():
        module.decoder.click_head.weight[0, -1] += 1e-6
    changed = parameter_fingerprints(module)
    assert changed['decoder.click_head.weight'] != initial['decoder.click_head.weight']
    assert changed['backbone.weight'] == initial['backbone.weight']


@pytest.mark.parametrize('field,value', [('load_only_weights', True), ('_manual_ckpt_path', 'teacher.ckpt'),
                                        ('_needs_trainer_reset', True), ('training_task', 'sid'),
                                        ('rerank_objective', 'pointwise')])
def test_fresh_audit_rejects_teacher_or_wrong_model(monkeypatch, field, value):
    module = model()
    state = trainer(module)
    torch.manual_seed(42)
    monkeypatch.setenv('PL_GLOBAL_SEED', '42')
    assert_fresh(state, module, 3, 42)
    setattr(module, field, value)
    with pytest.raises(AssertionError):
        assert_fresh(state, module, 3, 42)


@pytest.mark.parametrize('field,value', [('ckpt_path', 'resume.ckpt'), ('global_step', 1),
                                        ('current_epoch', 1), ('world_size', 1)])
def test_fresh_audit_rejects_restore(monkeypatch, field, value):
    module = model()
    state = trainer(module)
    torch.manual_seed(42)
    monkeypatch.setenv('PL_GLOBAL_SEED', '42')
    setattr(state, field, value)
    with pytest.raises(AssertionError):
        assert_fresh(state, module, 3, 42)


def test_fresh_audit_rejects_optimizer_state_and_wrong_seed(monkeypatch):
    module = model()
    state = trainer(module)
    torch.manual_seed(42)
    monkeypatch.setenv('PL_GLOBAL_SEED', '42')
    state.optimizers[0].state[next(module.parameters())] = {'step': torch.tensor(1.)}
    with pytest.raises(AssertionError, match='empty initial state'):
        assert_fresh(state, module, 3, 42)
    state.optimizers[0].state.clear()
    monkeypatch.setenv('PL_GLOBAL_SEED', '43')
    with pytest.raises(AssertionError, match='seed'):
        assert_fresh(state, module, 3, 42)


def test_updates_require_all_three_groups_and_actual_optimizer_steps():
    module = model()
    initial = parameter_fingerprints(module)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
    with pytest.raises(AssertionError, match='Missing actual'):
        update_evidence(module, optimizer, initial)
    for _ in range(2):
        for parameter in module.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    evidence = update_evidence(module, optimizer, initial)
    assert all(evidence['changed_parameters'][group] for group in GROUPS)
    assert set(evidence['optimizer_parameter_steps'].values()) == {2}
    next(iter(optimizer.state.values()))['step'].zero_()
    with pytest.raises(AssertionError, match='optimizer step'):
        update_evidence(module, optimizer, initial)


def test_callback_detects_parameter_change_between_constructor_and_fit(monkeypatch, tmp_path):
    module = model()
    state = trainer(module)
    torch.manual_seed(42)
    monkeypatch.setenv('PL_GLOBAL_SEED', '42')
    audit = RandomInitAudit(tmp_path)
    audit.setup(state, module, 'fit')
    with torch.no_grad():
        module.backbone.weight.add_(1)
    with pytest.raises(AssertionError, match='changed before training'):
        audit.on_fit_start(state, module)


def test_smoke_requires_real_update_and_validation(tmp_path):
    audit = RandomInitAudit(tmp_path, smoke=True)
    with pytest.raises(AssertionError, match='real validation'):
        audit.on_fit_end(SimpleNamespace(global_step=2), None)


def write_reports(directory):
    hashes = parameter_fingerprints(model())
    changed = {group: [name for name in hashes if
                      ('sid' if name.startswith('decoder.sid_head.') else
                       'click' if name.startswith('decoder.click_head.') else 'backbone') == group]
               for group in GROUPS}
    after = {name: dict(entry, sha256='f' * 64) for name, entry in hashes.items()}
    for rank in range(3):
        for phase in PHASES:
            report = dict(schema_version=1, status='passed', phase=phase,
                          method='joint-listwise-random-init', smoke=True,
                          global_rank=rank, world_size=3, seed=42,
                          global_step=0 if phase in ('initialization', 'train_start') else 2)
            if phase in ('initialization', 'train_start'):
                report.update(constructed_parameters_unchanged=True, optimizer_initially_empty=True,
                              initialization_checkpoint=None, resume_checkpoint=None,
                              parameter_fingerprints=hashes, checked_after_training_state_restore=True)
            elif phase == 'updates':
                report.update(gradient_checks=[dict.fromkeys(GROUPS, 1)] * 2,
                              changed_parameters=changed,
                              optimizer_parameter_steps=dict.fromkeys(hashes, 2),
                              parameter_fingerprints_after=after)
            else:
                report.update(validation_batches=2, first_two_updates_verified=True,
                              validation_metrics={'val/loss': 34., 'val/recall@5': 0., 'val/ndcg@5': 0.})
            (directory / f'rank_{rank}_{phase}.json').write_text(json.dumps(report))


def test_report_gate_requires_exact_three_ranks(tmp_path):
    write_reports(tmp_path)
    assert verify_reports(tmp_path, smoke=True)['status'] == 'passed'
    (tmp_path / 'rank_3_initialization.json').write_text('{}')
    with pytest.raises(AssertionError, match='rank audit reports'):
        verify_reports(tmp_path, smoke=True)


@pytest.mark.parametrize('phase,field,value', [
    ('initialization', 'global_rank', 1), ('initialization', 'resume_checkpoint', 'old.ckpt'),
    ('train_start', 'checked_after_training_state_restore', False),
    ('updates', 'optimizer_parameter_steps', {'p': 1}),
    ('updates', 'gradient_checks', [dict.fromkeys(GROUPS, 1)]),
    ('finished', 'global_step', 3), ('finished', 'validation_batches', 0),
    ('finished', 'validation_metrics', {'val/loss': float('nan'), 'val/recall@5': 0., 'val/ndcg@5': 0.}),
])
def test_report_gate_rejects_changed_evidence(tmp_path, phase, field, value):
    write_reports(tmp_path)
    path = tmp_path / f'rank_0_{phase}.json'
    report = json.loads(path.read_text())
    report[field] = value
    path.write_text(json.dumps(report))
    with pytest.raises(AssertionError):
        verify_reports(tmp_path, smoke=True)
