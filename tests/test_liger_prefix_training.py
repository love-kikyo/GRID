import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.baselines.liger_data import Record
from src.baselines.liger_full_catalog import RowCatalog
from src.baselines.liger_prefix_training import (
    check_progress, initial_progress, prefix_identity_digest, train_step,
)
from src.baselines.liger_standardization import StandardizedContent, fit_standardization
from src.baselines.liger_state import BudgetSchedule
from src.baselines.liger_supervision import expand_training_records, pack_supervised_batch


def fixture(*, attempts=3):
    torch.manual_seed(72)
    model = torch.nn.Module()
    model.emb_proj = torch.nn.Linear(3, 4)
    model.query = torch.nn.Parameter(torch.randn(7, 4))
    calls = []
    def forward(current, batch, device, *args):
        calls.append(batch)
        current.predicted_embedding = current.query[:len(batch['labels_sids'])]
        return SimpleNamespace(loss=current.predicted_embedding.square().mean()), None
    evaluation = SimpleNamespace(model_forward=forward)
    raw = np.random.default_rng(2).normal(size=(7, 3)).astype('float32')
    seen = np.array([0, 1, 2, 4, 6])
    content = StandardizedContent(raw, fit_standardization(raw, seen))
    sids = np.arange(28).reshape(7, 4)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = BudgetSchedule(optimizer, total=10, warmup=2)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    progress = initial_progress(max_optimizer_steps=10, max_attempts=attempts)
    examples = expand_training_records([
        Record(3, (1, 0, 2), 'a' * 64), Record(4, (2, 4), 'b' * 64)])
    return SimpleNamespace(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        evaluation=evaluation, sids=sids, content=content, seen=seen,
        catalog=RowCatalog(content, seen), progress=progress, examples=examples, calls=calls)


def step(f, **changes):
    arguments = dict(model=f.model, optimizer=f.optimizer, scheduler=f.scheduler, scaler=f.scaler,
        method={}, evaluation=f.evaluation, records=f.examples, sids=f.sids, content=f.content,
        seen=f.seen, catalog=f.catalog, progress=f.progress, chunk_size=2, expected_gradients=3)
    arguments.update(changes)
    return train_step(**arguments)


def test_same_parent_prefixes_are_distinct_equal_weight_targets_and_counted():
    f = fixture()
    result = step(f)
    assert result['prefix_count'] == result['rows'] == 3
    assert result['final_target_count'] == 2 and result['batch_unique_parents'] == 2
    assert result['prefix_identity_sha256'] == result['batch_identity_sha256']
    assert result['attempt'] == result['batch'] == result['updates'] == 1
    assert f.progress['prefix_count'] == f.progress['samples_in_epoch'] == 3
    assert f.progress['final_count'] == 2
    assert f.progress['attempts'] == f.progress['batches_consumed'] == 1
    assert f.progress['data_chain'] != '0' * 64 and not result['skipped']
    assert result['total_loss'] == pytest.approx(result['sid_loss'] + result['dense_loss'])
    assert result['learning_rate'] == .0003 / 2
    assert set(f.calls[0]) == {'input_sids', 'attention_mask_sids', 'labels_sids', 'input_embeddings'}
    np.testing.assert_array_equal(f.calls[0]['labels_sids'].numpy(), f.sids[[0, 2, 4]] + [1, 4097, 8193, 12289])
    assert torch.count_nonzero(f.calls[0]['input_embeddings'][0, 1:]) == 0


def test_real_joint_objective_update_matches_independent_monolithic_full_catalog():
    f = fixture()
    reference = copy.deepcopy(f.model)
    optimizer = torch.optim.AdamW(reference.parameters())
    scheduler = BudgetSchedule(optimizer, total=10, warmup=2)
    _, identity = pack_supervised_batch(f.examples, f.sids, f.content)
    query = reference.query[:len(f.examples)]
    vectors = reference.emb_proj(torch.from_numpy(f.content[f.seen]))
    logits = F.normalize(query, dim=-1) @ F.normalize(vectors, dim=-1).T / .07
    masked = query.new_full((len(query), len(f.content) - len(f.seen)), -100.)
    labels = torch.tensor(np.searchsorted(f.seen, identity['target_rows']))
    reference_loss = query.square().mean() + F.cross_entropy(torch.cat([logits, masked], dim=1), labels)
    reference_loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
    optimizer.step()
    scheduler.step()
    result = step(f)
    assert result['total_loss'] == pytest.approx(reference_loss.item(), rel=1e-6)
    for actual, expected in zip(f.model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_prefix_identity_includes_target_position_and_order_not_just_parent_id():
    f = fixture()
    _, first = pack_supervised_batch(f.examples[:1], f.sids, f.content)
    _, second = pack_supervised_batch(f.examples[1:2], f.sids, f.content)
    assert first['parent_user_ids'].tolist() == second['parent_user_ids'].tolist()
    assert first['parent_serialized_sha256'] == second['parent_serialized_sha256']
    assert prefix_identity_digest(first) != prefix_identity_digest(second)
    _, forward = pack_supervised_batch(f.examples, f.sids, f.content)
    _, reverse = pack_supervised_batch(f.examples[::-1], f.sids, f.content)
    assert prefix_identity_digest(forward) != prefix_identity_digest(reverse)


class FakeAmpScaler:
    """CPU-only skip accounting fixture; intentionally not a production scaler."""
    def __init__(self): self.factor = 128.
    def get_scale(self): return self.factor
    def scale(self, loss): return loss
    def unscale_(self, optimizer): pass
    def step(self, optimizer):
        self.skipped = any(not torch.isfinite(parameter.grad).all()
                           for group in optimizer.param_groups for parameter in group['params']
                           if parameter.grad is not None)
        if not self.skipped: optimizer.step()
    def update(self):
        if self.skipped: self.factor /= 2


def test_amp_skip_consumes_prefixes_and_attempt_but_not_schedule_or_update():
    f = fixture(attempts=2)
    f.scaler = FakeAmpScaler()
    before = copy.deepcopy(f.model.state_dict())
    hook = f.model.query.register_hook(lambda gradient: torch.full_like(gradient, float('inf')))
    skipped = step(f)
    hook.remove()
    assert skipped['skipped'] and skipped['grad_norm'] is None
    assert f.progress['updates'] == f.scheduler.updates == 0
    assert f.progress['attempts'] == f.progress['skipped_updates'] == 1
    assert f.progress['prefix_count'] == 3 and f.progress['final_count'] == 2
    for name, value in before.items(): torch.testing.assert_close(f.model.state_dict()[name], value)
    accepted = step(f)
    assert not accepted['skipped'] and accepted['attempt'] == 2 and accepted['updates'] == 1
    assert f.progress['prefix_count'] == 6 and f.progress['final_count'] == 4
    assert f.scheduler.updates == 1
    with pytest.raises(ValueError, match='budget exhausted'): step(f)


def test_attempt_budget_can_stop_before_the_fixed_optimizer_schedule():
    f = fixture(attempts=1)
    step(f)
    assert f.scheduler.total == 10 and f.scheduler.updates == 1
    with pytest.raises(ValueError, match='budget exhausted'): step(f)
    assert len(f.calls) == 1


def test_optimizer_budget_also_stops_when_attempt_cap_is_larger():
    f = fixture(attempts=20)
    f.scheduler = BudgetSchedule(f.optimizer, total=2, warmup=1)
    f.progress = initial_progress(max_optimizer_steps=2, max_attempts=20)
    step(f)
    step(f)
    with pytest.raises(ValueError, match='budget exhausted'): step(f)


def test_progress_fixed_budgets_allow_steps_based_validation_and_epoch_census():
    budget = dict(max_optimizer_steps=100, max_attempts=3, batch_size=4, prefixes_per_epoch=10)
    p = initial_progress(max_optimizer_steps=100, max_attempts=3)
    check_progress(p, budget)
    p.update(next_batch=3, samples_in_epoch=10, batches_consumed=3, attempts=3,
             prefix_count=10, final_count=4, updates=2, skipped_updates=1,
             validations=[{'attempt': 1}, {'attempt': 3}])
    check_progress(p, budget)
    p.update(epoch=1, next_batch=0, samples_in_epoch=0)
    check_progress(p, budget)
    p['prefix_count'] = 11
    with pytest.raises(ValueError, match='census'): check_progress(p, budget)


@pytest.mark.parametrize('key,value', [('attempts', 1), ('prefix_count', -1),
    ('final_count', 1), ('samples_in_epoch', 1), ('next_batch', 1), ('updates', 1),
    ('data_chain', 'not-sha'), ('validations', 0), ('epoch', True), ('max_attempts', 99)])
def test_bad_cursor_is_rejected(key, value):
    p = initial_progress(max_optimizer_steps=10, max_attempts=3)
    p[key] = value
    with pytest.raises(ValueError):
        check_progress(p, dict(max_optimizer_steps=10, max_attempts=3))


def test_schedule_changes_bad_gt_duplicate_prefix_and_catalog_subsetting_fail():
    f = fixture()
    f.scheduler.total = 11
    with pytest.raises(ValueError, match='schedule'): step(f)
    f.scheduler.total = 10
    with pytest.raises(ValueError, match='Duplicate supervised'):
        step(f, records=[f.examples[0], f.examples[0]])
    missing_gt = expand_training_records([Record(9, (1, 3), 'e' * 64)])
    with pytest.raises(ValueError, match='missing'): step(f, records=missing_gt)
    with pytest.raises(ValueError, match='complete frozen'):
        step(f, catalog=RowCatalog(f.content, f.seen[:3]))
    with pytest.raises(ValueError, match='complete frozen'):
        step(f, catalog=RowCatalog(np.zeros(f.content.shape, dtype=np.float32), f.seen))
    assert not f.calls and f.progress['attempts'] == 0


def test_old_progress_without_fixed_prefix_budget_cannot_silently_resume():
    from src.baselines.liger_training import initial_progress as old_progress
    with pytest.raises(ValueError):
        check_progress(old_progress(), dict(max_optimizer_steps=10, max_attempts=3))


def test_original_full_state_resume_retains_prefix_cursor_and_requires_new_binding(tmp_path):
    from src.baselines.liger_state import capture, restore_checkpoint, save_checkpoint

    uninterrupted = fixture()
    step(uninterrupted)
    binding = dict(supervision_tag='bounded_last8_all_prefix_equal_weight_v1',
                   standardization_sha256='a' * 64, standardization_stats_sha256='b' * 64,
                   standardization_seen_rows_sha256='c' * 64)
    checkpoint = tmp_path / 'prefix-step1.pt'
    save_checkpoint(checkpoint, capture(uninterrupted.model, uninterrupted.optimizer,
        uninterrupted.scheduler, uninterrupted.scaler, uninterrupted.progress, binding))
    expected = step(uninterrupted)
    resumed = fixture()
    resumed.progress = restore_checkpoint(checkpoint, resumed.model, resumed.optimizer,
        resumed.scheduler, resumed.scaler, binding)
    actual = step(resumed)
    assert actual == expected and resumed.progress == uninterrupted.progress
    for a, b in zip(resumed.model.parameters(), uninterrupted.model.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    with pytest.raises(ValueError, match='binding'):
        restore_checkpoint(checkpoint, resumed.model, resumed.optimizer, resumed.scheduler,
                           resumed.scaler, dict(binding, standardization_sha256='d' * 64))
    with pytest.raises(ValueError, match='binding'):
        restore_checkpoint(checkpoint, resumed.model, resumed.optimizer, resumed.scheduler,
                           resumed.scaler, {'purpose': 'old-final-only'})
