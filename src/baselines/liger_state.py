"""Fixed-budget scheduling and complete, create-only LIGER checkpoints."""
import hashlib
import math
import os
from pathlib import Path
import random

import numpy as np
import torch


def tree_digest(value):
    """Canonical value digest, independent of torch.save archive metadata."""
    h = hashlib.sha256()
    def visit(obj):
        if isinstance(obj, torch.Tensor):
            t = obj.detach().cpu().contiguous()
            h.update(str((str(t.dtype), tuple(t.shape))).encode())
            h.update(t.numpy().tobytes())
        elif isinstance(obj, np.ndarray):
            h.update(str((str(obj.dtype), obj.shape)).encode())
            h.update(obj.tobytes())
        elif isinstance(obj, dict):
            h.update(b'dict')
            for key in sorted(obj, key=repr):
                visit(key)
                visit(obj[key])
        elif isinstance(obj, (tuple, list)):
            h.update(type(obj).__name__.encode())
            for part in obj: visit(part)
        else:
            h.update((type(obj).__name__ + ':' + repr(obj) + '\n').encode())
    visit(value)
    return h.hexdigest()


class BudgetSchedule:
    """LR for the next successful update; AMP skips do not advance the budget."""
    def __init__(self, optimizer, *, total=19260, warmup=963, base_lr=.0003):
        if not 0 < warmup < total or base_lr <= 0:
            raise ValueError('Invalid fixed training budget')
        self.optimizer = optimizer
        self.total, self.warmup, self.base_lr = total, warmup, base_lr
        self.updates = 0
        self._apply()

    def _apply(self):
        step = min(self.updates + 1, self.total)
        scale = step / self.warmup if step <= self.warmup else .5 * (
            1 + math.cos(math.pi * (step - self.warmup) / (self.total - self.warmup)))
        for group in self.optimizer.param_groups: group['lr'] = self.base_lr * scale

    def step(self):
        if self.updates >= self.total:
            raise ValueError('Original training budget exhausted')
        self.updates += 1
        self._apply()

    def state_dict(self):
        return dict(total=self.total, warmup=self.warmup, base_lr=self.base_lr, updates=self.updates)

    def load_state_dict(self, state):
        if any(state[k] != getattr(self, k) for k in ('total', 'warmup', 'base_lr')):
            raise ValueError('Cannot change the original schedule on resume')
        if not 0 <= state['updates'] <= self.total:
            raise ValueError('Invalid update cursor')
        self.updates = state['updates']
        self._apply()


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA RNG device count changed')
        torch.cuda.set_rng_state_all(state['cuda'])


def capture(model, optimizer, scheduler, scaler, progress, binding):
    if scheduler.updates != progress['updates']:
        raise ValueError('Schedule and optimizer progress disagree')
    return dict(format_version=1, binding=binding, model=model.state_dict(),
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(), rng=rng_state(), progress=progress,
                model_training=model.training)


def save_checkpoint(path, state):
    """Write a new checkpoint atomically; never replace an existing artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('xb') as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    # Atomic create-only publication on the same filesystem.
    os.link(temporary, path)
    temporary.unlink()


def restore_checkpoint(path, model, optimizer, scheduler, scaler, binding):
    # Only local, hash-bound checkpoints produced by this adapter are accepted.
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['format_version'] != 1 or state['binding'] != binding:
        raise ValueError('Checkpoint protocol/source binding mismatch')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    scaler.load_state_dict(state['scaler'])
    if scheduler.updates != state['progress']['updates']:
        raise ValueError('Invalid saved update count')
    model.train(state['model_training'])
    restore_rng(state['rng'])
    return state['progress']


def better_validation(metrics, epoch, best):
    """Frozen selection: hybrid Recall@5, NDCG@5, then earlier epoch."""
    key = (metrics['Recall@5'], metrics['NDCG@5'], -epoch)
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in key[:2]):
        raise ValueError('Invalid validation metrics')
    return best is None or key > (best['Recall@5'], best['NDCG@5'], -best['epoch'])
