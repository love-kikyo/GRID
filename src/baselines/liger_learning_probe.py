"""CPU-only helpers for bounded LIGER learning diagnostics, not evaluation.

Fit and reserved parents must both come from the frozen training split. The
reserved group is a diagnostic holdout, never the official validation/test.
"""
from numbers import Integral

import numpy as np

from src.baselines.liger_supervision import PrefixExample, expand_training_records


def _integer(value, name, minimum=1):
    if (not isinstance(value, Integral) or isinstance(value, (bool, np.bool_))
            or value < minimum):
        raise ValueError(f'{name} must be an integer >= {minimum}')


def split_parents(records, fit_count=128, reserved_count=128, seed=20261022):
    """Randomly split exactly fit_count + reserved_count unique training users.

    Stable input ordering and seed reproduce both groups, including group order.
    No parent is discarded or shared. This helper never reads any split file.
    """
    for name, value in (('fit_count', fit_count), ('reserved_count', reserved_count)):
        _integer(value, name)
    _integer(seed, 'seed', minimum=0)
    records = list(records)
    if len(records) != fit_count + reserved_count:
        raise ValueError('Parent count must equal fit_count + reserved_count')
    expand_training_records(records, mode='final_only')  # Validate identity/sequence.
    order = np.random.default_rng(seed).permutation(len(records))
    return ([records[i] for i in order[:fit_count]],
            [records[i] for i in order[fit_count:]])


def diagnostic_catalog(fit_records, seen_rows, size=4096, seed=20261022):
    """Include all fit suffix-window rows, then uniformly sample training seen.

    Only fit records are accepted: there is deliberately no reserved/heldout
    label argument. ``seen_rows`` must be a sorted unique training-seen index.
    Returns sorted unique int64 rows. Required fit rows must fit within ``size``
    and all must already belong to seen; membership is never repaired silently.
    A sample of at most ``size`` seen positions avoids shuffling the full index.
    """
    _integer(size, 'size')
    _integer(seed, 'seed', minimum=0)
    seen = np.asarray(seen_rows)
    if (seen.ndim != 1 or not np.issubdtype(seen.dtype, np.integer)
            or len(seen) < size or (seen < 0).any()
            or (seen[1:] <= seen[:-1]).any()
            or (seen > np.iinfo(np.int64).max).any()):
        raise ValueError('seen_rows must be sorted unique nonnegative rows and cover size')
    parents = expand_training_records(fit_records, history_items=7, mode='final_only')
    required = np.unique(np.asarray([row for p in parents for row in p.sequence], dtype=np.int64))
    if len(required) > size:
        raise ValueError('Required fit rows exceed diagnostic catalog size')
    positions = np.searchsorted(seen, required)
    if (positions >= len(seen)).any() or not np.array_equal(seen[positions], required):
        raise ValueError('All fit suffix-window rows must belong to training seen')
    selected = set(required.tolist())
    if len(selected) < size:
        sampled = np.random.default_rng(seed).choice(len(seen), size=size, replace=False)
        for index in sampled:
            selected.add(int(seen[index]))
            if len(selected) == size:
                break
    if len(selected) != size:
        raise AssertionError('Diagnostic catalog fill did not reach the requested size')
    return np.asarray(sorted(selected), dtype=np.int64)


def deterministic_batches(examples, batch_size=32, steps=1024, seed=42):
    """Yield exactly ``steps`` full lists of examples; shuffle each pool epoch.

    Sampling is without replacement within an epoch. The incomplete tail is
    dropped each epoch; a single batch therefore never repeats a prefix. The
    same example may recur after reshuffling the pool. Callers must record the
    actual consumed identities/is_final counts: equal steps do not imply equal
    final-GT exposure between final-only and expanded-prefix conditions.
    """
    for name, value in (('batch_size', batch_size), ('steps', steps)):
        _integer(value, name)
    _integer(seed, 'seed', minimum=0)
    examples = list(examples)
    if len(examples) < batch_size:
        raise ValueError('Example pool must contain at least batch_size examples')
    identities, parents = set(), {}
    for example in examples:
        if not isinstance(example, PrefixExample):
            raise ValueError('Expected PrefixExample instances')
        identity = (example.parent_user_id, example.prefix_target_index)
        if identity in identities:
            raise ValueError('Duplicate supervised prefix in example pool')
        identities.add(identity)
        parent_sha = parents.setdefault(example.parent_user_id, example.parent_serialized_sha256)
        if parent_sha != example.parent_serialized_sha256:
            raise ValueError('Conflicting parent identity in example pool')
    rng, emitted = np.random.default_rng(seed), 0
    usable = len(examples) - len(examples) % batch_size
    while emitted < steps:
        order = rng.permutation(len(examples))
        for start in range(0, usable, batch_size):
            yield [examples[i] for i in order[start:start + batch_size]]
            emitted += 1
            if emitted == steps:
                return


def generation_metrics(tokens, labels):
    """Natural beam-order binary recall/NDCG and cumulative-prefix hit counts.

    Inputs use the same token offsets: tokens[B,20,4], labels[B,4]. A duplicated
    GT counts once at its first beam rank; duplicate_gt_queries makes that
    policy observable. Prefix counts refer to at least one of the *final* 20
    beams matching the first p levels, not intermediate search-tree coverage.
    """
    tokens, labels = np.asarray(tokens), np.asarray(labels)
    if (tokens.ndim != 3 or tokens.shape[1:] != (20, 4) or len(tokens) < 1
            or labels.shape != (len(tokens), 4)
            or not np.issubdtype(tokens.dtype, np.integer)
            or not np.issubdtype(labels.dtype, np.integer)
            or (tokens < 0).any() or (labels <= 0).any()):
        raise ValueError('Expected nonnegative integer tokens[B,20,4] and positive labels[B,4]')
    matches = tokens == labels[:, None, :]
    prefixes = np.logical_and.accumulate(matches, axis=2)
    hits = prefixes[:, :, -1]
    first_rank = np.where(hits.any(axis=1), hits.argmax(axis=1) + 1, 21)
    result = {
        'num_examples': len(tokens),
        'prefix_hit_counts': prefixes.any(axis=1).sum(axis=0).astype(int).tolist(),
        'duplicate_gt_queries': int((hits.sum(axis=1) > 1).sum()),
        'duplicate_policy': 'binary_first_hit',
    }
    for k in (5, 10, 20):
        found = first_rank <= k
        result[f'Recall@{k}'] = float(found.mean())
        result[f'NDCG@{k}'] = float(np.where(found, 1.0 / np.log2(first_rank + 1), 0).mean())
    return result
