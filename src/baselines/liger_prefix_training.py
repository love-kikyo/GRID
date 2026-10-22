"""Equal-prefix SID + exact full-seen dense training with fixed step budgets.

Every PrefixExample supplies the same four SID labels and one dense GT. A parent
may occur repeatedly with different prefix positions; parent IDs are provenance
only. Neither the objective nor the catalog is replaced by a diagnostic subset.
"""
import hashlib
from numbers import Integral
import re

import numpy as np
import torch

from src.baselines.liger_full_catalog import full_catalog_cross_entropy
from src.baselines.liger_state import tree_digest
from src.baselines.liger_supervision import pack_supervised_batch


def _integer(value, *, positive=False):
    return (isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
            and value >= (1 if positive else 0))


def initial_progress(*, max_optimizer_steps, max_attempts):
    if not _integer(max_optimizer_steps, positive=True) or not _integer(max_attempts, positive=True):
        raise ValueError('Explicit positive update and attempt budgets are required')
    return dict(epoch=0, next_batch=0, samples_in_epoch=0, batches_consumed=0,
                attempts=0, prefix_count=0, final_count=0, updates=0, skipped_updates=0,
                max_optimizer_steps=int(max_optimizer_steps), max_attempts=int(max_attempts),
                data_chain='0' * 64, best=None, validations=[])


def check_progress(progress, budget):
    """Validate a prefix cursor without assuming epochs or validations coincide.

    ``samples_in_epoch`` and ``prefix_count`` count supervised prefixes, not
    unique parents. Optional ``batch_size`` / ``prefixes_per_epoch`` strengthen
    cursor checks once a full training-only census has established the latter.
    """
    counters = ('epoch', 'next_batch', 'samples_in_epoch', 'batches_consumed', 'attempts',
                'prefix_count', 'final_count', 'updates', 'skipped_updates')
    if any(key not in progress or not _integer(progress[key]) for key in counters):
        raise ValueError('Inconsistent prefix training/data cursor counters')
    for key in ('max_optimizer_steps', 'max_attempts'):
        if (key not in budget or not _integer(budget[key], positive=True)
                or not _integer(progress.get(key), positive=True) or progress[key] != budget[key]):
            raise ValueError('Cannot change the fixed training budget')
    p = progress
    if (p['attempts'] != p['batches_consumed']
            or p['updates'] + p['skipped_updates'] != p['attempts']
            or p['updates'] > p['max_optimizer_steps'] or p['attempts'] > p['max_attempts']
            or p['next_batch'] > p['attempts'] or p['samples_in_epoch'] > p['prefix_count']
            or p['prefix_count'] < p['attempts'] or p['final_count'] > p['prefix_count']
            or p['attempts'] - p['next_batch'] < p['epoch']
            or p['prefix_count'] - p['samples_in_epoch'] < p['epoch']
            or (p['epoch'] == 0 and (p['attempts'] != p['next_batch']
                                     or p['prefix_count'] != p['samples_in_epoch']))
            or (p['next_batch'] == 0) != (p['samples_in_epoch'] == 0)
            or not isinstance(p.get('validations'), list)
            or not isinstance(p.get('data_chain'), str)
            or re.fullmatch('[0-9a-f]{64}', p['data_chain']) is None):
        raise ValueError('Inconsistent prefix training/data cursor')
    if 'batch_size' in budget:
        size = budget['batch_size']
        if not _integer(size, positive=True):
            raise ValueError('Invalid prefix batch size')
        if p['next_batch'] and not (p['next_batch'] - 1) * size < p['samples_in_epoch'] <= p['next_batch'] * size:
            raise ValueError('Prefix sample cursor disagrees with consumed batches')
    if 'prefixes_per_epoch' in budget:
        count = budget['prefixes_per_epoch']
        if not _integer(count, positive=True) or 'batch_size' not in budget:
            raise ValueError('A prefix census requires positive count and batch size')
        batches = (count + budget['batch_size'] - 1) // budget['batch_size']
        if (p['next_batch'] > batches
                or p['samples_in_epoch'] != min(p['next_batch'] * budget['batch_size'], count)
                or p['prefix_count'] != p['epoch'] * count + p['samples_in_epoch']
                or p['attempts'] != p['epoch'] * batches + p['next_batch']):
            raise ValueError('Prefix cursor disagrees with the frozen epoch census')


def prefix_identity_digest(identity):
    """Ordered identity binds parent serialization AND the supervised position."""
    keys = ('parent_user_ids', 'parent_serialized_sha256', 'parent_sequence_lengths',
            'window_start_indices', 'prefix_target_indices', 'parent_target_indices',
            'parent_prefix_counts', 'is_final', 'target_rows', 'history_rows')
    return tree_digest({key: identity[key] for key in keys})


def train_step(model, optimizer, scheduler, scaler, method, evaluation, records,
               sids, content, seen, catalog, progress, *, chunk_size=65536, expected_gradients=152):
    """Consume one prefix batch; AMP skips consume attempts/data, not updates."""
    budget = {key: progress.get(key) for key in ('max_optimizer_steps', 'max_attempts')}
    check_progress(progress, budget)
    if (not model.training or scheduler.updates != progress['updates']
            or scheduler.total != progress['max_optimizer_steps']):
        raise ValueError('Expected training mode and the original aligned schedule')
    if progress['attempts'] >= progress['max_attempts'] or scheduler.updates >= scheduler.total:
        raise ValueError('Training attempt/update budget exhausted')
    if (len(catalog) != len(seen) or getattr(catalog, 'content', None) is not content
            or getattr(catalog, 'rows', None) is not seen):
        raise ValueError('Training requires the same content and complete frozen seen RowCatalog')
    device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    records = list(records)
    inputs, identity = pack_supervised_batch(records, sids, content, weighting='prefix')
    if identity['loss_weighting'] != 'prefix' or not np.all(identity['loss_weights'] == 1.):
        raise ValueError('Formal prefix loss must weight every supervised prefix equally')
    positions = np.searchsorted(seen, identity['target_rows'])
    if (positions >= len(seen)).any() or not np.array_equal(seen[positions], identity['target_rows']):
        raise ValueError('Training prefix GT missing from the full seen catalog')
    data_sha = prefix_identity_digest(identity)
    labels = torch.as_tensor(positions, dtype=torch.long, device=device)
    batch = {key: torch.from_numpy(value) for key, value in inputs.items()}
    outputs, _ = evaluation.model_forward(model, batch, device, 4, method)
    dense_loss = full_catalog_cross_entropy(
        model.predicted_embedding.float(), labels, model.emb_proj, catalog,
        chunk_size=chunk_size, temperature=.07, masked_count=len(content) - len(seen))
    loss = outputs.loss + dense_loss
    if not torch.isfinite(loss).item():
        raise ValueError('Nonfinite prefix training loss')
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if len(gradients) != expected_gradients:
        raise ValueError(f'Expected {expected_gradients} gradient tensors, got {len(gradients)}')
    finite = all(torch.isfinite(gradient).all().item() for gradient in gradients)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.) if finite else None
    if norm is not None and not torch.isfinite(norm).item():
        raise ValueError('Nonfinite gradient norm')
    learning_rate, old_scale = optimizer.param_groups[0]['lr'], scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    skipped = scaler.get_scale() < old_scale
    if skipped == finite:
        raise ValueError('AMP skip disagrees with gradient finiteness')
    if not skipped:
        scheduler.step()
        progress['updates'] += 1
    else:
        progress['skipped_updates'] += 1
    if len(optimizer.state) != expected_gradients and progress['updates']:
        raise ValueError('Optimizer state coverage mismatch')
    if any(int(state['step'].item()) != progress['updates'] for state in optimizer.state.values()):
        raise ValueError('Optimizer update counters diverged')
    final_count = int(identity['is_final'].sum())
    progress['data_chain'] = hashlib.sha256((progress['data_chain'] + data_sha).encode()).hexdigest()
    progress['next_batch'] += 1
    progress['batches_consumed'] += 1
    progress['attempts'] += 1
    progress['samples_in_epoch'] += len(records)
    progress['prefix_count'] += len(records)
    progress['final_count'] += final_count
    check_progress(progress, budget)
    return dict(
        batch=progress['next_batch'], rows=len(records), attempt=progress['attempts'],
        prefix_count=len(records), final_target_count=final_count,
        batch_unique_parents=len(np.unique(identity['parent_user_ids'])),
        prefix_identity_sha256=data_sha, batch_identity_sha256=data_sha,
        first_user=int(identity['parent_user_ids'][0]), last_user=int(identity['parent_user_ids'][-1]),
        updates=progress['updates'], skipped=skipped, learning_rate=learning_rate,
        scale_before=old_scale, scale_after=scaler.get_scale(),
        sid_loss=outputs.loss.item(), dense_loss=dense_loss.item(), total_loss=loss.item(),
        gradient_tensors=len(gradients), grad_norm=norm.item() if norm is not None else None)
