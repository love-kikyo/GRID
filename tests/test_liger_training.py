from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.baselines.liger_data import Record
from src.baselines.liger_full_catalog import RowCatalog
from src.baselines.liger_state import BudgetSchedule
from src.baselines.liger_training import check_progress, initial_progress, train_step


def test_shared_update_visits_full_catalog_and_counts_real_records():
    model = torch.nn.Module()
    model.emb_proj = torch.nn.Linear(3, 4)
    model.query = torch.nn.Parameter(torch.randn(2, 4))
    def forward(m, batch, device, *args):
        m.predicted_embedding = m.query
        return SimpleNamespace(loss=m.query.square().mean()), None
    evaluation = SimpleNamespace(model_forward=forward)
    content = np.random.default_rng(1).normal(size=(7, 3)).astype('float32')
    seen = np.array([0, 1, 2, 4, 6])
    sids = np.arange(28).reshape(7, 4)
    opt = torch.optim.AdamW(model.parameters())
    sched = BudgetSchedule(opt, total=10, warmup=2)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    p = initial_progress()
    rows = [Record(3, (1, 0), 'a' * 64), Record(4, (2, 4), 'b' * 64)]
    r = train_step(model, opt, sched, scaler, {}, evaluation, rows, sids, content,
                   seen, RowCatalog(content, seen), p, chunk_size=2, expected_gradients=3)
    assert r['rows'] == 2 and p['next_batch'] == p['updates'] == 1
    assert p['samples_in_epoch'] == 2 and p['data_chain'] != '0' * 64
    assert r['learning_rate'] == .0003 / 2 and not r['skipped']


def test_progress_rejects_drift_and_retains_tail():
    budget = dict(batches_per_epoch=3, max_epochs=3, batch_size=4,
                  records_per_epoch=10, max_optimizer_steps=9)
    p = initial_progress()
    check_progress(p, budget)
    p.update(next_batch=3, samples_in_epoch=10, batches_consumed=3, updates=2, skipped_updates=1)
    check_progress(p, budget)
    p['samples_in_epoch'] = 12
    with pytest.raises(ValueError, match='cursor'): check_progress(p, budget)
