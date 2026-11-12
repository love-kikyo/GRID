"""One sampled candidate pool and projection draw per logical optimizer batch.

This is an isolated accumulation adapter, not a replacement for historical
sampled training. The full logical batch supplies the forced-positive union.
Candidate projection happens ONCE, before query microbatches, retaining its
autograd graph (including the same dropout draw) until the last backward.
Query-side dropout draws may differ with microbatch size; no historical RNG
trajectory or floating-point bitwise equivalence is claimed.

Call ``microbatch_dense_loss`` for an exact, nonoverlapping cover of the logical
batch. Each returned loss is already divided by the FULL logical batch size.
Multiply each microbatch's mean SID loss by ``pool.microbatch_weight(...)``.
Backward with ``retain_graph=True`` except on the final microbatch, release
query/output references, then clip, AMP-step, optimizer-step and scheduler-step
only once. A pool is ephemeral and MUST NOT survive an optimizer update.
"""
from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np
import torch
import torch.nn.functional as F

from src.baselines.liger_sampled_catalog import CandidateSample, sample_candidates


LOGICAL_BATCH_VERSION = 'shared_logical_positive_union_projection_once_v1'


def _integer(value, minimum=0):
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)) and value >= minimum


def _projection_state(projection):
    return (tuple((id(p), p._version, p.dtype, p.device, p.requires_grad)
                  for p in projection.parameters()),
            tuple((id(m), m.training) for m in projection.modules()))


@dataclass(frozen=True)
class LogicalCandidatePool:
    """Transient graph plus immutable identity for one logical batch only."""
    sample: CandidateSample
    items: torch.Tensor
    target_indices: torch.Tensor
    log_inclusion: torch.Tensor
    logical_size: int
    temperature: float
    masked_count: int
    projection: torch.nn.Module
    projection_state: tuple

    def microbatch_weight(self, start, stop):
        if (not _integer(start) or not _integer(stop, 1)
                or not 0 <= start < stop <= self.logical_size):
            raise ValueError('Microbatch must be a nonempty slice of the logical batch')
        return (stop - start) / self.logical_size

    def check_current(self):
        if _projection_state(self.projection) != self.projection_state:
            raise ValueError('Logical candidate pool cannot cross a parameter, mode or device change')


def prepare_candidate_pool(projection, catalog, labels, *, num_negatives, seed,
                           attempt=0, temperature=.07, masked_count=0):
    """Sample/project once using ALL logical-batch labels, not each microbatch.

    Positions and labels index the frozen train-seen catalog, not global asset
    rows. The existing sampler retains its correction and RNG/attempt policy.
    Only frozen input feature rows may be cached; projected items stay attached
    to their trainable projection graph throughout this logical batch.
    """
    if not isinstance(projection, torch.nn.Module):
        raise ValueError('Expected a trainable projection module')
    parameters = tuple(projection.parameters())
    if (not parameters or not any(p.requires_grad for p in parameters)
            or any(True for _ in projection.buffers())):
        raise ValueError('Expected trainable projection parameters and no stateful buffers')
    reference = parameters[0]
    if (reference.dtype not in (torch.float32, torch.float64)
            or reference.device.type not in ('cpu', 'cuda')
            or any(p.dtype != reference.dtype or p.device != reference.device for p in parameters)):
        raise ValueError('Projection parameters must share FP32/FP64 dtype and CPU/CUDA device')
    if not torch.is_grad_enabled() or torch.is_autocast_enabled(reference.device.type):
        raise ValueError('Prepare candidate graph with gradients enabled and outside autocast')
    if (not isinstance(labels, torch.Tensor) or labels.dtype != torch.long
            or labels.ndim != 1 or not len(labels) or labels.device != reference.device):
        raise ValueError('Expected nonempty logical long labels on the projection device')
    if (not isinstance(temperature, Real) or isinstance(temperature, (bool, np.bool_))
            or not math.isfinite(temperature) or temperature <= 0 or not _integer(masked_count)):
        raise ValueError('Invalid temperature or masked population')
    if isinstance(catalog, torch.Tensor) and catalog.requires_grad:
        raise ValueError('Catalog input features must be frozen')
    sample = sample_candidates(labels, len(catalog), num_negatives, seed, attempt)
    raw = catalog[sample.positions]
    if isinstance(raw, torch.Tensor) and raw.requires_grad:
        raise ValueError('Catalog input features must be frozen')
    raw = torch.as_tensor(raw, dtype=reference.dtype, device=reference.device)
    if (raw.ndim != 2 or raw.shape[0] != len(sample.positions) or not raw.shape[1]
            or not torch.isfinite(raw).all().item()):
        raise ValueError('Expected finite sampled feature rows')
    before = _projection_state(projection)
    projected = projection(raw)
    if (not isinstance(projected, torch.Tensor) or projected.ndim != 2
            or projected.shape[0] != len(sample.positions) or not projected.shape[1]
            or projected.dtype != reference.dtype or projected.device != reference.device
            or not projected.requires_grad or not torch.isfinite(projected).all().item()
            or _projection_state(projection) != before):
        raise ValueError('Expected finite differentiable projected items with unchanged projection')
    return LogicalCandidatePool(
        sample=sample, items=F.normalize(projected, dim=-1),
        target_indices=torch.as_tensor(sample.target_indices, dtype=torch.long, device=reference.device),
        log_inclusion=torch.as_tensor(sample.log_inclusion, dtype=reference.dtype, device=reference.device),
        logical_size=len(labels), temperature=float(temperature), masked_count=int(masked_count),
        projection=projection, projection_state=before)


def microbatch_dense_loss(query, pool, start, stop):
    """Return this slice's corrected CE SUM / full logical batch size.

    Do not average this result again or divide by the number of microbatches.
    Tail logical batches (e.g. 81 prefixes) use their actual logical_size.
    The caller owns nonoverlapping coverage and optimizer/scaler lifecycle.
    """
    if not isinstance(pool, LogicalCandidatePool):
        raise ValueError('Expected a prepared logical candidate pool')
    pool.microbatch_weight(start, stop)
    pool.check_current()
    if (not isinstance(query, torch.Tensor) or query.shape != (stop - start, pool.items.shape[1])
            or query.dtype != pool.items.dtype or query.device != pool.items.device
            or not torch.isfinite(query).all().item()):
        raise ValueError('Query must match the logical slice, projection width, dtype and device')
    if not torch.is_grad_enabled() or torch.is_autocast_enabled(query.device.type):
        raise ValueError('Compute dense loss with gradients enabled and outside autocast')
    logits = F.normalize(query, dim=-1) @ pool.items.T / pool.temperature
    logits = logits - pool.log_inclusion
    if not torch.isfinite(logits).all().item():
        raise ValueError('Nonfinite sampled logits')
    log_z = torch.logsumexp(logits, dim=1)
    if pool.masked_count:
        log_z = torch.logaddexp(log_z, log_z.new_tensor(math.log(pool.masked_count) - 100.))
    targets = pool.target_indices[start:stop]
    return (log_z - logits.gather(1, targets[:, None]).squeeze(1)).sum() / pool.logical_size
