"""Real CPU logical-step integration: accumulation, AMP, cursor and replay."""
import copy
import gc
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from scripts.liger_capacity_runtime import logical_step, new_progress
from src.baselines.liger_data import Record
from src.baselines.liger_full_catalog import RowCatalog
from src.baselines.liger_sampled_catalog import sampled_catalog_cross_entropy
from src.baselines.liger_state import BudgetSchedule, capture, restore_checkpoint, save_checkpoint, tree_digest
from src.baselines.liger_supervision import expand_training_records, pack_supervised_batch


class ToyModel(nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.encoder = nn.Linear(5, 4)
        self.emb_proj = nn.Sequential(nn.Linear(5, 4), nn.Dropout(dropout))
        self.predicted_embedding = None


class OfficialForward:
    def __init__(self):
        self.micro_sizes = []
        self.input_graph_refs = []
        self.all_previous_released = []

    def model_forward(self, model, batch, device, depth, method):
        self.all_previous_released.append(all(r() is None for r in self.input_graph_refs))
        self.micro_sizes.append(len(batch['labels_sids']))
        features = batch['input_embeddings'].to(device).sum(dim=1)
        inputs_embeds = model.emb_proj(features)
        self.input_graph_refs.append(weakref.ref(inputs_embeds))
        query = model.encoder(features) + inputs_embeds
        model.predicted_embedding = query
        loss = F.cross_entropy(query, batch['labels_sids'][:, 0].to(device) % 4)
        return SimpleNamespace(loss=loss), {'inputs_embeds': inputs_embeds}


def make_context(dropout=0.):
    torch.manual_seed(854)
    model = ToyModel(dropout).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.035)
    scheduler = BudgetSchedule(optimizer, total=22460, warmup=1123, base_lr=3e-4)
    scaler = torch.amp.GradScaler('cpu', init_scale=128., growth_interval=1000)
    content = np.random.default_rng(41).normal(size=(23, 5)).astype(np.float32)
    seen = np.arange(19, dtype=np.int32)
    sids = (np.arange(23 * 4).reshape(23, 4) % 4096).astype(np.int32)
    return SimpleNamespace(model=model, optimizer=optimizer, scheduler=scheduler,
        scaler=scaler, official=OfficialForward(), method={}, content=content,
        seen=seen, sids=sids, catalog=RowCatalog(content, seen),
        binding={'purpose': 'cpu_fixture_only'})


def examples(count):
    parents = [Record(i, (i % 19, (i + 7) % 19), f'{i:064x}') for i in range(count)]
    return expand_training_records(parents, mode='final_only')


def monolithic_update(ctx, records, attempt=0):
    """Independent orchestration using the historical single sampled CE call."""
    ctx.optimizer.zero_grad(set_to_none=True)
    inputs, identity = pack_supervised_batch(records, ctx.sids, ctx.content)
    batch = {k: torch.from_numpy(v) for k, v in inputs.items()}
    outputs, _ = ctx.official.model_forward(ctx.model, batch, torch.device('cpu'), 4, {})
    labels = torch.tensor(np.searchsorted(ctx.seen, identity['target_rows']), dtype=torch.long)
    dense = sampled_catalog_cross_entropy(ctx.model.predicted_embedding, labels,
        ctx.model.emb_proj, ctx.catalog, num_negatives=32768, seed=42, attempt=attempt,
        masked_count=len(ctx.content) - len(ctx.seen))
    loss = outputs.loss + dense
    ctx.scaler.scale(loss).backward()
    ctx.scaler.unscale_(ctx.optimizer)
    torch.nn.utils.clip_grad_norm_(ctx.model.parameters(), 1.)
    ctx.scaler.step(ctx.optimizer); ctx.scaler.update(); ctx.scheduler.step()
    ctx.model.predicted_embedding = None
    return outputs.loss.item(), dense.item()


@pytest.mark.parametrize('count,micro', [(1024, 512), (513, 512), (81, 512), (1024, 1024)])
def test_actual_logical_step_matches_monolithic_all_parameters_and_optimizer(count, micro):
    ctx, control = make_context(), make_context()
    records, progress = examples(count), new_progress()
    sid, dense = monolithic_update(control, records)
    result = logical_step(ctx, records, progress, micro, expected_gradients=4)
    assert result['sid_loss'] == pytest.approx(sid, abs=1e-6)
    assert result['dense_loss'] == pytest.approx(dense, abs=3e-6)
    for actual, expected in zip(ctx.model.parameters(), control.model.parameters()):
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
        for key in ('exp_avg', 'exp_avg_sq', 'step'):
            torch.testing.assert_close(ctx.optimizer.state[actual][key],
                                       control.optimizer.state[expected][key], atol=1e-7, rtol=1e-5)
    assert ctx.scheduler.state_dict() == control.scheduler.state_dict()
    assert progress['attempts'] == progress['updates'] == progress['next_batch'] == 1
    assert progress['prefix_count'] == progress['final_count'] == count
    assert result['candidate_classes'] == 19 and result['gradient_tensors'] == 4
    assert result['microbatches'] == len(ctx.official.micro_sizes)
    assert sum(ctx.official.micro_sizes) == count
    gc.collect()
    assert all(ctx.official.all_previous_released)
    assert all(ref() is None for ref in ctx.official.input_graph_refs)
    assert ctx.model.predicted_embedding is None


def test_one_sampling_projection_clip_and_amp_update_per_logical_batch(monkeypatch):
    import src.baselines.liger_logical_batch as pool_module
    calls = dict(prepare=0, sample=0, clip=0, scale=0, unscale_=0, step=0, update=0)
    ctx, records, p = make_context(), examples(2048), new_progress()
    def spy(owner, name, key):
        original = getattr(owner, name)
        def wrapped(*args, **kwargs):
            calls[key] += 1
            return original(*args, **kwargs)
        monkeypatch.setattr(owner, name, wrapped)
    spy(pool_module, 'prepare_candidate_pool', 'prepare')
    spy(pool_module, 'sample_candidates', 'sample')
    spy(torch.nn.utils, 'clip_grad_norm_', 'clip')
    for name in ('scale', 'unscale_', 'step', 'update'): spy(ctx.scaler, name, name)
    result = logical_step(ctx, records, p, 512, expected_gradients=4)
    assert result['microbatches'] == 4
    assert calls == dict(prepare=1, sample=1, clip=1, scale=4, unscale_=1, step=1, update=1)
    assert {int(s['step']) for s in ctx.optimizer.state.values()} == {1}


def test_amp_skip_consumes_one_logical_attempt_without_advancing_weights_or_schedule():
    ctx, records, p = make_context(), examples(513), new_progress()
    logical_step(ctx, records, p, 512, expected_gradients=4)
    before_model = copy.deepcopy(ctx.model.state_dict())
    before_schedule = copy.deepcopy(ctx.scheduler.state_dict())
    before_chain = p['data_chain']
    parameter = next(ctx.model.parameters())
    handle = parameter.register_hook(lambda gradient: torch.full_like(gradient, float('inf')))
    result = logical_step(ctx, records, p, 512, expected_gradients=4)
    handle.remove()
    assert result['skipped'] and result['scale_after'] == result['scale_before'] / 2
    assert result['grad_norm'] is None
    assert p['attempts'] == p['next_batch'] == 2 and p['updates'] == 1
    assert p['skipped_updates'] == 1 and p['prefix_count'] == 1026
    assert p['data_chain'] != before_chain
    assert ctx.scheduler.state_dict() == before_schedule
    for key, value in ctx.model.state_dict().items(): torch.testing.assert_close(value, before_model[key], atol=0, rtol=0)
    assert {int(s['step']) for s in ctx.optimizer.state.values()} == {1}
    recovered = logical_step(ctx, records, p, 512, expected_gradients=4)
    assert not recovered['skipped'] and p['updates'] == 2 and p['attempts'] == 3


def test_new_context_checkpoint_replay_matches_full_state_with_dropout(tmp_path):
    ctx, records, p = make_context(dropout=.3), examples(513), new_progress()
    logical_step(ctx, records, p, 512, expected_gradients=4)
    path = tmp_path / 'step01.pt'
    save_checkpoint(path, capture(ctx.model, ctx.optimizer, ctx.scheduler, ctx.scaler, p, ctx.binding))
    expected = logical_step(ctx, records, p, 512, expected_gradients=4)
    expected_state = tree_digest(capture(ctx.model, ctx.optimizer, ctx.scheduler, ctx.scaler, p, ctx.binding))
    resumed = make_context(dropout=.3)
    restored = restore_checkpoint(path, resumed.model, resumed.optimizer, resumed.scheduler,
                                  resumed.scaler, resumed.binding)
    actual = logical_step(resumed, records, restored, 512, expected_gradients=4)
    actual_state = tree_digest(capture(resumed.model, resumed.optimizer, resumed.scheduler,
                                     resumed.scaler, restored, resumed.binding))
    assert actual == expected and actual_state == expected_state
