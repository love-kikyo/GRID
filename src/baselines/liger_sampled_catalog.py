"""Shared, without-replacement sampled dense loss for the fixed seen catalog.

All distinct batch targets are included with probability one. K other classes
are sampled uniformly from their complement, with inclusion probability K/R.
Subtracting log(K/R) from sampled logits gives an unbiased estimator of the
softmax partition (conditional on the batch), NOT an unbiased loss or gradient.
Sampling the whole complement recovers full-catalog CE. Sampling never changes
the inference catalog or the weights of the supervised prefix examples.
"""
from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np
import torch
import torch.nn.functional as F


SAMPLING_VERSION = 'uniform_complement_without_replacement_ht_v1'


def _integer(value, *, minimum=0, bounded=True):
    return (isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
            and value >= minimum and (not bounded or value <= np.iinfo(np.int64).max))


def sampling_binding(config):
    """Canonical objective identity, stored with the prefix training cursor."""
    if (not isinstance(config, dict) or set(config) != {'num_negatives', 'seed'}
            or not _integer(config['num_negatives'], minimum=1)
            or not _integer(config['seed'])):
        raise ValueError('Sampling config requires exactly positive num_negatives and nonnegative seed')
    return dict(sampling_version=SAMPLING_VERSION, num_negatives=int(config['num_negatives']),
                seed=int(config['seed']), temperature=.07)


@dataclass(frozen=True)
class CandidateSample:
    """Indices are positions in the ordered train-seen catalog, not asset IDs."""
    positions: np.ndarray
    target_indices: np.ndarray
    log_inclusion: np.ndarray
    population_size: int
    forced_positives: int
    sampled_negatives: int
    remaining_population: int


def sample_candidates(labels, catalog_size, num_negatives, seed, attempt=0):
    """Stateless CPU sampling with O(batch + K) storage, no catalog permutation.

    ``attempt`` is the pre-step data-attempt counter, including AMP-skipped
    batches. The sampler has a private PCG64 stream and does not advance model,
    data-shuffle or global NumPy RNG state. Multi-rank training is not wired by
    this module; any future runner must bind its rank/seed policy explicitly.
    """
    if (not _integer(catalog_size, minimum=1)
            or not _integer(num_negatives, minimum=1)
            or not _integer(seed) or not _integer(attempt)):
        raise ValueError('Expected positive catalog/negative counts and nonnegative seed/attempt')
    if (not isinstance(labels, torch.Tensor) or labels.dtype != torch.long
            or labels.ndim != 1 or not len(labels)):
        raise ValueError('Expected nonempty one-dimensional long labels')
    targets = labels.detach().cpu().numpy()
    if (targets < 0).any() or (targets >= catalog_size).any():
        raise ValueError('Labels must index the complete train-seen catalog')
    positives = np.unique(targets)
    remaining = int(catalog_size - len(positives))
    count = min(int(num_negatives), remaining)
    if count == remaining:
        positions = np.arange(catalog_size, dtype=np.int64)
        log_inclusion = np.zeros(catalog_size, dtype=np.float64)
    else:
        # Fixed-width entropy avoids ambiguous concatenation of variable-width
        # seed/attempt integers by SeedSequence (e.g. 2**32 versus two words).
        entropy = [int(seed) & 0xffffffff, int(seed) >> 32,
                   int(attempt) & 0xffffffff, int(attempt) >> 32]
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
        # Floyd sampling: a uniform subset without a population-sized buffer,
        # including when K is small relative to tens of millions of classes.
        chosen = set()
        for upper in range(remaining - count, remaining):
            candidate = int(rng.integers(0, upper + 1))
            chosen.add(upper if candidate in chosen else candidate)
        ranks = np.asarray(sorted(chosen), dtype=np.int64)
        # Rank-select in the complement of sorted positives; repeated adjusted
        # boundaries are intentional (consecutive forced-positive positions).
        negatives = ranks + np.searchsorted(
            positives - np.arange(len(positives), dtype=np.int64), ranks, side='right')
        positions = np.sort(np.concatenate((positives, negatives)))
        log_inclusion = np.zeros(len(positions), dtype=np.float64)
        log_inclusion[np.searchsorted(positions, negatives)] = math.log(count / remaining)
    target_indices = np.searchsorted(positions, targets)
    return CandidateSample(positions, target_indices, log_inclusion, int(catalog_size),
                           len(positives), count, remaining)


def sampled_catalog_cross_entropy(query, labels, projection, catalog, *, num_negatives,
                                 seed, attempt=0, temperature=.07, masked_count=0):
    """Mean corrected sampled CE, with gradients through query and projection.

    Only sampled/forced rows are fetched and projected. Frozen input features
    may be raw tensors, RowCatalog or CachedSeenCatalog; learned projections
    are never cached/detached. A single projected vector is shared by duplicate
    item identities. Dropout is allowed, but its draw layout need not match the
    original chunked full loss. Exact full-loss tests disable dropout.
    """
    if (not isinstance(query, torch.Tensor) or query.ndim != 2 or not all(query.shape)
            or query.dtype not in (torch.float32, torch.float64)
            or query.device.type not in ('cpu', 'cuda')):
        raise ValueError('Expected nonempty FP32/FP64 query matrix on CPU/CUDA')
    if torch.is_autocast_enabled(query.device.type):
        raise ValueError('Sampled dense loss must run outside autocast')
    if (not isinstance(labels, torch.Tensor) or labels.shape != (len(query),)
            or labels.device != query.device):
        raise ValueError('Labels must match query batch and device')
    if (isinstance(temperature, (bool, np.bool_)) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature <= 0
            or not _integer(masked_count, bounded=False)):
        raise ValueError('Invalid temperature or masked count')
    if not torch.isfinite(query).all().item():
        raise ValueError('Nonfinite query')
    if isinstance(catalog, torch.Tensor) and catalog.requires_grad:
        raise ValueError('Catalog input features must be frozen')
    parameters = tuple(projection.parameters())
    if (not parameters or not any(p.requires_grad for p in parameters)
            or any(p.dtype != query.dtype or p.device != query.device for p in parameters)):
        raise ValueError('Trainable projection and query must share dtype/device')
    if any(True for _ in projection.buffers()):
        raise ValueError('Stateful projection buffers are unsupported')
    sample = sample_candidates(labels, len(catalog), num_negatives, seed, attempt)
    raw = catalog[sample.positions]
    if isinstance(raw, torch.Tensor) and raw.requires_grad:
        raise ValueError('Catalog input features must be frozen')
    raw = torch.as_tensor(raw, device=query.device, dtype=query.dtype)
    if raw.ndim != 2 or len(raw) != len(sample.positions) or not torch.isfinite(raw).all().item():
        raise ValueError('Expected finite sampled feature rows')
    items = projection(raw)
    if items.shape != (len(sample.positions), query.shape[1]) or not torch.isfinite(items).all().item():
        raise ValueError('Expected finite projected items matching query dimension')
    logits = (F.normalize(query, dim=-1) @ F.normalize(items, dim=-1).T) / temperature
    logits = logits - torch.as_tensor(sample.log_inclusion, device=query.device, dtype=query.dtype)
    if not torch.isfinite(logits).all().item():
        raise ValueError('Nonfinite sampled logits')
    targets = torch.as_tensor(sample.target_indices, device=query.device, dtype=torch.long)
    log_z = torch.logsumexp(logits, dim=1)
    if masked_count:
        log_z = torch.logaddexp(log_z, log_z.new_tensor(math.log(masked_count) - 100.))
    return (log_z - logits.gather(1, targets[:, None]).squeeze(1)).mean()
