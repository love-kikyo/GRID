"""Single-GPU training update shared by the smoke and formal entry points."""
import hashlib
import numpy as np
import torch

from src.baselines.liger_data import pack_batch
from src.baselines.liger_full_catalog import full_catalog_cross_entropy
from src.baselines.liger_state import tree_digest


def initial_progress():
    return dict(epoch=0, next_batch=0, samples_in_epoch=0, batches_consumed=0,
                updates=0, skipped_updates=0, data_chain='0' * 64, best=None, validations=[])


def check_progress(p, budget):
    epoch, batch = p['epoch'], p['next_batch']
    per_epoch = budget['batches_per_epoch']
    if (not 0 <= epoch <= budget['max_epochs'] or not 0 <= batch <= per_epoch
            or p['batches_consumed'] != epoch * per_epoch + batch
            or p['updates'] + p['skipped_updates'] != p['batches_consumed']
            or p['samples_in_epoch'] != min(batch * budget['batch_size'], budget['records_per_epoch'])
            or len(p['validations']) != epoch or len(p['data_chain']) != 64
            or p['updates'] > budget['max_optimizer_steps']):
        raise ValueError('Inconsistent training/data cursor')


def train_step(model, optimizer, scheduler, scaler, method, evaluation, records,
               sids, content, seen, catalog, progress, *, chunk_size=65536, expected_gradients=152):
    if not model.training or scheduler.updates != progress['updates']:
        raise ValueError('Expected training mode and aligned schedule')
    if scheduler.updates >= scheduler.total:
        raise ValueError('Training update budget exhausted')
    device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    inputs, identity = pack_batch(records, sids, content)
    positions = np.searchsorted(seen, identity['target_rows'])
    if (positions >= len(seen)).any() or not np.array_equal(seen[positions], identity['target_rows']):
        raise ValueError('Training GT missing from seen catalog')
    labels = torch.as_tensor(positions, dtype=torch.long, device=device)
    batch = {k: torch.from_numpy(v) for k, v in inputs.items()}
    outputs, _ = evaluation.model_forward(model, batch, device, 4, method)
    dense_loss = full_catalog_cross_entropy(model.predicted_embedding.float(), labels,
        model.emb_proj, catalog, chunk_size=chunk_size, temperature=.07,
        masked_count=len(content) - len(seen))
    loss = outputs.loss + dense_loss
    if not torch.isfinite(loss).item(): raise ValueError('Nonfinite training loss')
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if len(gradients) != expected_gradients:
        raise ValueError(f'Expected {expected_gradients} gradient tensors, got {len(gradients)}')
    finite = all(torch.isfinite(g).all().item() for g in gradients)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.) if finite else None
    if norm is not None and not torch.isfinite(norm).item(): raise ValueError('Nonfinite gradient norm')
    lr, old_scale = optimizer.param_groups[0]['lr'], scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    skipped = scaler.get_scale() < old_scale
    if skipped == finite: raise ValueError('AMP skip disagrees with gradient finiteness')
    if not skipped:
        scheduler.step()
        progress['updates'] += 1
    else:
        progress['skipped_updates'] += 1
    if len(optimizer.state) != expected_gradients and progress['updates']:
        raise ValueError('Optimizer state coverage mismatch')
    if any(int(s['step'].item()) != progress['updates'] for s in optimizer.state.values()):
        raise ValueError('Optimizer update counters diverged')
    data_sha = tree_digest(dict(user_ids=identity['user_ids'], serialized_sha256=identity['serialized_sha256']))
    progress['data_chain'] = hashlib.sha256((progress['data_chain'] + data_sha).encode()).hexdigest()
    progress['next_batch'] += 1
    progress['batches_consumed'] += 1
    progress['samples_in_epoch'] += len(records)
    return dict(batch=progress['next_batch'], rows=len(records), batch_identity_sha256=data_sha,
                first_user=int(identity['user_ids'][0]), last_user=int(identity['user_ids'][-1]),
                updates=progress['updates'], skipped=skipped, learning_rate=lr,
                scale_before=old_scale, scale_after=scaler.get_scale(),
                sid_loss=outputs.loss.item(), dense_loss=dense_loss.item(), total_loss=loss.item(),
                gradient_tensors=len(gradients), grad_norm=norm.item() if norm is not None else None)
