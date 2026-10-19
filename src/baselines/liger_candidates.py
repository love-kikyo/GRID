"""Exact SID-to-row identity and GT-free candidate sanitation.

This module does not choose seen/cold policy, constrain generation, or backfill
invalid beams. Row zero is a real item; -1 is the sole missing-row sentinel.
"""
import numpy as np


def sid_keys(codes):
    codes = np.asarray(codes)
    if codes.ndim < 1 or codes.shape[-1] != 4 or not np.issubdtype(codes.dtype, np.integer):
        raise ValueError('Expected four integer SID codes')
    if (codes < 0).any() or (codes >= 4096).any():
        raise ValueError('SID code outside [0,4096)')
    result = np.zeros(codes.shape[:-1], dtype=np.uint64)
    for level in range(4):
        result = (result << np.uint64(12)) | codes[..., level].astype(np.uint64)
    return result


class SidRowIndex:
    """One-to-one index over the entire frozen asset, with exact unknown checks."""
    def __init__(self, sorted_keys, rows):
        self.keys, self.rows = np.asarray(sorted_keys), np.asarray(rows)
        if (self.keys.ndim != 1 or self.rows.shape != self.keys.shape or not len(self.keys)
                or self.keys.dtype != np.uint64 or self.rows.dtype != np.int32):
            raise ValueError('Expected nonempty uint64 keys and int32 rows')
        if self.keys[-1] >= 2**48 or not (self.keys[1:] > self.keys[:-1]).all():
            raise ValueError('SID aliases or unsorted/out-of-range keys')
        if not np.array_equal(np.sort(self.rows), np.arange(len(self.rows), dtype=np.int32)):
            raise ValueError('Index must contain every asset row exactly once')

    @classmethod
    def build(cls, semantic_ids, chunk_size=262144):
        if (semantic_ids.ndim != 2 or semantic_ids.shape[1] != 4 or not len(semantic_ids)
                or len(semantic_ids) > np.iinfo(np.int32).max or chunk_size < 1):
            raise ValueError('Invalid asset shape or chunk size')
        keys = np.empty(len(semantic_ids), dtype=np.uint64)
        for start in range(0, len(keys), chunk_size):
            keys[start:start + chunk_size] = sid_keys(semantic_ids[start:start + chunk_size])
        order = np.argsort(keys, kind='stable')
        return cls(keys[order], order.astype(np.int32))

    def lookup_codes(self, codes):
        keys = sid_keys(codes)
        positions = np.searchsorted(self.keys, keys)
        safe = np.minimum(positions, len(self.keys) - 1)
        valid = (positions < len(self.keys)) & (self.keys[safe] == keys)
        return np.where(valid, self.rows[safe], -1).astype(np.int32)

    def lookup_tokens(self, tokens):
        """Undo T5 offsets; padding, EOS and cross-codebook tokens stay missing."""
        tokens = np.asarray(tokens)
        if tokens.ndim < 1 or tokens.shape[-1] != 4 or not np.issubdtype(tokens.dtype, np.integer):
            raise ValueError('Expected four integer generated tokens')
        offsets = np.asarray([1, 4097, 8193, 12289], dtype=np.int64)
        valid_codes = (tokens >= offsets) & (tokens < offsets + 4096)
        raw = np.where(valid_codes, tokens, offsets).astype(np.int64) - offsets
        rows = self.lookup_codes(raw)
        return np.where(valid_codes.all(axis=-1), rows, -1).astype(np.int32)


def unique_beam_rows(rows):
    """Keep the first occurrence at its original rank; no extra beam requests."""
    rows = np.asarray(rows)
    if (rows.ndim != 2 or not np.issubdtype(rows.dtype, np.integer) or (rows < -1).any()
            or (rows > np.iinfo(np.int32).max).any()):
        raise ValueError('Expected beam rows with -1 for missing candidates')
    result = rows.copy()
    for batch in result:
        seen = set()
        for position, row in enumerate(batch):
            if row < 0 or row in seen:
                batch[position] = -1
            else:
                seen.add(int(row))
    return result


def rank_unique_rows(rows, scores, k):
    """Dense-only ranking, deterministic ties by row ID; no GT argument.

    Duplicate rows retain their maximum score. Suitable for merging exact
    chunk top-k results and beam candidates, but does not compute those scores.
    """
    rows, scores = np.asarray(rows), np.asarray(scores)
    if (rows.ndim != 1 or scores.shape != rows.shape or k < 1
            or not np.issubdtype(rows.dtype, np.integer) or (rows < -1).any()
            or (rows > np.iinfo(np.int32).max).any()):
        raise ValueError('Invalid candidate rows/scores/k')
    valid = rows >= 0
    if not np.isfinite(scores[valid]).all():
        raise ValueError('Nonfinite score for valid candidate')
    best = {}
    for row, score in zip(rows[valid], scores[valid]):
        best[int(row)] = max(best.get(int(row), -np.inf), float(score))
    ordered = sorted(best, key=lambda row: (-best[row], row))[:k]
    output_rows, output_scores = np.full(k, -1, dtype=np.int32), np.full(k, -np.inf, dtype=np.float64)
    output_rows[:len(ordered)] = ordered
    output_scores[:len(ordered)] = [best[row] for row in ordered]
    return output_rows, output_scores
