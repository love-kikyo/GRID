"""Bounded train-only prefix supervision without changing frozen data assets.

This module opens no files and does not expand validation/test records. Callers
must supply records from the frozen *training* stream. Expansion is deliberately
bounded to the final ``history_items + 1`` items, not the full stored sequence.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from src.baselines.liger_inputs import pack_sequences


@dataclass(frozen=True)
class PrefixExample:
    """One supervised prefix; all indices are zero-based item positions.

    ``prefix_target_index`` is relative to the retained suffix window, while
    ``parent_target_index`` addresses the original record. The original final
    GT is retained exactly once and identified by ``is_final``. A user's ID is
    provenance only and must never be supplied to the model.
    """

    parent_user_id: int
    parent_serialized_sha256: str
    parent_sequence_length: int
    window_start_index: int
    prefix_target_index: int
    sequence: tuple
    parent_prefix_count: int
    is_final: bool

    @property
    def parent_target_index(self):
        return self.window_start_index + self.prefix_target_index


def _positive_integer(value, label):
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value < 1:
        raise ValueError(f'{label} must be a positive integer')


def _check_identity(user_id, digest):
    if (not isinstance(user_id, Integral) or isinstance(user_id, (bool, np.bool_))
            or not np.iinfo(np.int64).min <= user_id <= np.iinfo(np.int64).max):
        raise ValueError('Parent user ID must be an int64 integer')
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in '0123456789abcdefABCDEF' for c in digest)):
        raise ValueError('Parent serialized SHA256 must contain 64 hexadecimal characters')


def _check_sequence(sequence):
    if (not isinstance(sequence, (tuple, list)) or not 2 <= len(sequence) <= 1000
            or any(not isinstance(row, Integral) or isinstance(row, (bool, np.bool_))
                   or row < 0 or row > np.iinfo(np.int64).max for row in sequence)):
        raise ValueError('Invalid history-last row sequence')


def expand_training_records(records, history_items=7, mode='all_prefix'):
    """Expand a finite batch of distinct training parents in deterministic order.

    ``all_prefix`` emits target positions 1..n-1 within each bounded suffix,
    shortest prefix first. ``final_only`` emits only its original final target
    and reproduces the existing input packing. Duplicate parent user IDs are
    rejected here; repeated items inside a sequence are intentionally preserved.
    This function does not relabel or rewrite the original records.
    """
    _positive_integer(history_items, 'history_items')
    if mode not in {'all_prefix', 'final_only'}:
        raise ValueError('mode must be all_prefix or final_only')
    records = list(records)
    if not records:
        raise ValueError('Nonempty training parents required')
    seen_users, result = set(), []
    for record in records:
        _check_identity(record.user_id, record.serialized_sha256)
        _check_sequence(record.sequence)
        if record.user_id in seen_users:
            raise ValueError('Duplicate parent user IDs in training records')
        seen_users.add(record.user_id)
        start = max(0, len(record.sequence) - history_items - 1)
        window = tuple(int(row) for row in record.sequence[start:])
        targets = range(1, len(window)) if mode == 'all_prefix' else (len(window) - 1,)
        for target in targets:
            result.append(PrefixExample(
                parent_user_id=int(record.user_id),
                parent_serialized_sha256=record.serialized_sha256,
                parent_sequence_length=len(record.sequence),
                window_start_index=start, prefix_target_index=target,
                sequence=window[:target + 1], parent_prefix_count=len(targets),
                is_final=target == len(window) - 1,
            ))
    return result


def _check_examples(examples, history_items):
    identities, parents, longest = set(), {}, {}
    for example in examples:
        if not isinstance(example, PrefixExample):
            raise ValueError('Expected PrefixExample instances')
        _check_identity(example.parent_user_id, example.parent_serialized_sha256)
        _check_sequence(example.sequence)
        for label in ('parent_sequence_length', 'prefix_target_index', 'parent_prefix_count'):
            _positive_integer(getattr(example, label), label)
        parent_length = example.parent_sequence_length
        expected_start = max(0, parent_length - history_items - 1)
        if (not isinstance(example.window_start_index, Integral)
                or isinstance(example.window_start_index, (bool, np.bool_))
                or example.window_start_index != expected_start or parent_length > 1000
                or example.prefix_target_index != len(example.sequence) - 1
                or len(example.sequence) > history_items + 1
                or example.parent_target_index >= parent_length
                or not isinstance(example.is_final, (bool, np.bool_))
                or example.is_final != (example.parent_target_index == parent_length - 1)):
            raise ValueError('Inconsistent prefix window, target position, or final GT identity')
        possible_prefixes = parent_length - expected_start - 1
        if (example.parent_prefix_count not in {1, possible_prefixes}
                or (example.parent_prefix_count == 1 and not example.is_final)):
            raise ValueError('Inconsistent parent prefix count')
        parent = (example.parent_serialized_sha256, parent_length,
                  example.window_start_index, example.parent_prefix_count)
        previous = parents.setdefault(example.parent_user_id, parent)
        if parent != previous:
            raise ValueError('Conflicting identity for repeated parent user ID')
        identity = (example.parent_user_id, example.prefix_target_index)
        if identity in identities:
            raise ValueError('Duplicate supervised prefix identity')
        identities.add(identity)
        previous_sequence = longest.get(example.parent_user_id, ())
        overlap = min(len(previous_sequence), len(example.sequence))
        if tuple(previous_sequence[:overlap]) != tuple(example.sequence[:overlap]):
            raise ValueError('Conflicting prefix sequences for the same parent')
        if len(example.sequence) > len(previous_sequence):
            longest[example.parent_user_id] = example.sequence


def pack_supervised_batch(examples, semantic_ids, content, history_items=7,
                          codebook_size=4096, weighting='prefix'):
    """Return ``(model_inputs, metadata)`` with provenance/weights kept separate.

    ``prefix`` (default) gives each example weight 1. ``parent`` gives weight
    1 / the parent's selected prefix count; over a complete expansion each
    parent contributes total weight 1. For a complete batch the weighted mean
    is ``sum(loss_i * weight_i) / sum(weight_i)``. When splitting a parent's
    prefixes over optimizer microbatches, callers must retain that normalization
    across the complete accumulation group; averaging microbatch means is not
    generally parent-uniform. This function does not apply or normalize a loss.

    Subsets of an expansion are allowed, but duplicate prefixes and conflicting
    provenance are rejected. Inputs retain the existing right-padding scheme.
    """
    _positive_integer(history_items, 'history_items')
    _positive_integer(codebook_size, 'codebook_size')
    if weighting not in {'prefix', 'parent'}:
        raise ValueError('weighting must be prefix or parent')
    examples = list(examples)
    if not examples:
        raise ValueError('Nonempty supervised examples required')
    _check_examples(examples, history_items)
    inputs = pack_sequences([e.sequence for e in examples], semantic_ids, content,
                            history_items=history_items, codebook_size=codebook_size)
    counts = np.asarray([e.parent_prefix_count for e in examples], dtype=np.int64)
    metadata = {
        'parent_user_ids': np.asarray([e.parent_user_id for e in examples], dtype=np.int64),
        'parent_serialized_sha256': [e.parent_serialized_sha256 for e in examples],
        'parent_sequence_lengths': np.asarray([e.parent_sequence_length for e in examples], dtype=np.int64),
        'window_start_indices': np.asarray([e.window_start_index for e in examples], dtype=np.int64),
        'prefix_target_indices': np.asarray([e.prefix_target_index for e in examples], dtype=np.int64),
        'parent_target_indices': np.asarray([e.parent_target_index for e in examples], dtype=np.int64),
        'is_final': np.asarray([e.is_final for e in examples], dtype=np.bool_),
        'parent_prefix_counts': counts,
        'loss_weights': (1.0 / counts).astype(np.float32) if weighting == 'parent'
                        else np.ones(len(examples), dtype=np.float32),
        'loss_weighting': weighting,
        'target_rows': inputs.pop('target_rows'),
        'history_rows': inputs.pop('history_rows'),
    }
    return inputs, metadata
