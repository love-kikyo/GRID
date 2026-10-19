import copy
import random

import numpy as np
import pytest
import torch

from src.baselines.liger_state import (BudgetSchedule, better_validation, capture, restore_checkpoint,
                                     save_checkpoint, tree_digest)


def components():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.2), torch.nn.Linear(4, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=.0003)
    sched = BudgetSchedule(opt, total=20, warmup=3)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    return model, opt, sched, scaler


def update(parts, progress):
    model, opt, sched, scaler = parts
    opt.zero_grad(set_to_none=True)
    x = torch.randn(8, 3) + float(np.random.random()) + random.random()
    loss = model(x).square().mean()
    loss.backward()
    opt.step()
    sched.step()
    progress['updates'] += 1
    progress['next_batch'] += 1


def test_resume_matches_continuous_model_optimizer_schedule_and_rng(tmp_path):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    parts = components()
    progress = dict(updates=0, next_batch=0)
    update(parts, progress)
    path = tmp_path / 'step1.pt'
    save_checkpoint(path, capture(*parts, progress, {'protocol': 'fixed'}))
    update(parts, progress)
    expected = tree_digest(capture(*parts, progress, {'protocol': 'fixed'}))
    other = components()
    restored = restore_checkpoint(path, *other, {'protocol': 'fixed'})
    update(other, restored)
    assert tree_digest(capture(*other, restored, {'protocol': 'fixed'})) == expected


def test_checkpoint_is_create_only_and_rejects_different_binding(tmp_path):
    parts = components()
    path = tmp_path / 'state.pt'
    save_checkpoint(path, capture(*parts, {'updates': 0}, {'protocol': 1}))
    with pytest.raises(FileExistsError): save_checkpoint(path, {})
    with pytest.raises(ValueError, match='binding'):
        restore_checkpoint(path, *parts, {'protocol': 2})


def test_fixed_schedule_and_skipped_updates():
    parts = components()
    _, opt, sched, _ = parts
    assert opt.param_groups[0]['lr'] == pytest.approx(.0003 / 3)
    state = copy.deepcopy(sched.state_dict())
    assert sched.state_dict() == state  # no step on AMP skip
    sched.step()
    sched.step()
    assert opt.param_groups[0]['lr'] == .0003
    for _ in range(18): sched.step()
    assert opt.param_groups[0]['lr'] == 0
    with pytest.raises(ValueError, match='exhausted'): sched.step()
    changed = dict(state, total=30)
    with pytest.raises(ValueError, match='original'): sched.load_state_dict(changed)


def test_selection_only_uses_frozen_validation_priority():
    best = dict(epoch=1, **{'Recall@5': .2, 'NDCG@5': .1})
    assert better_validation({'Recall@5': .21, 'NDCG@5': .09}, 2, best)
    assert better_validation({'Recall@5': .2, 'NDCG@5': .11}, 2, best)
    assert not better_validation({'Recall@5': .2, 'NDCG@5': .1, 'Recall@20': 1.}, 2, best)
    with pytest.raises(ValueError): better_validation({'Recall@5': float('nan'), 'NDCG@5': .1}, 2, best)


def test_digest_sensitive_to_contents_not_mapping_order():
    a = dict(a=torch.tensor([1.]), b=np.array([2]))
    assert tree_digest(a) == tree_digest(dict(reversed(list(a.items()))))
    a['a'][0] = 3
    assert tree_digest(a) != tree_digest(dict(a=torch.tensor([1.]), b=np.array([2])))
