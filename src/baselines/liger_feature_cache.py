"""Immutable RAM cache of fixed train-seen standardized INPUT features only.

No learned projection, loss, GPU tensor or complete asset matrix is cached.
Construction retains one FP32 seen matrix plus bounded temporary chunks. The
original source and seen row identities must remain immutable for the lifetime
of the cache, as with the existing frozen RowCatalog contract.
"""
import hashlib
from numbers import Integral
import re
import time
from types import MappingProxyType

import numpy as np

from src.baselines.liger_standardization import (
    SCHEMA, StandardizedContent, statistics_digest,
)


_BINDING_KEYS = frozenset({'standardization_stats_sha256', 'seen_rows_sha256', 'source_sha256'})


def _positive_integer(value):
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)) and value > 0


def _binding(binding):
    if not isinstance(binding, dict) or set(binding) != _BINDING_KEYS:
        raise ValueError('Expected exact statistics/seen/source SHA256 binding keys')
    if any(not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None
           for value in binding.values()):
        raise ValueError('Cache bindings must be lowercase SHA256 digests')
    return dict(binding)


def _seen_hash(rows, chunk_size):
    digest = hashlib.sha256()
    for start in range(0, len(rows), chunk_size):
        digest.update(np.asarray(rows[start:start + chunk_size], dtype='<i8').tobytes(order='C'))
    return digest.hexdigest()


def _validate_source(content, seen, expected_binding, chunk_size):
    if not isinstance(content, StandardizedContent):
        raise ValueError('Cache input must be the original StandardizedContent, not raw or already cached content')
    if not _positive_integer(chunk_size):
        raise ValueError('chunk_size must be a positive integer')
    if (not isinstance(seen, np.ndarray) or seen.ndim != 1 or not len(seen)
            or seen.dtype != np.dtype('int32')):
        raise ValueError('Expected sorted unique int32 seen rows within the original asset')
    for start in range(0, len(seen), chunk_size):
        rows = seen[start:start + chunk_size]
        if (rows[0] < 0 or rows[-1] >= len(content)
                or not (rows[1:] > rows[:-1]).all()
                or (start and rows[0] <= seen[start - 1])):
            raise ValueError('Expected sorted unique int32 seen rows within the original asset')
    binding = _binding(expected_binding)
    if (content.stats_sha256 != binding['standardization_stats_sha256']
            or content.content_sha256 != binding['source_sha256']
            or _seen_hash(seen, chunk_size) != binding['seen_rows_sha256']):
        raise ValueError('Standardized content or seen rows disagree with the cache binding')
    # Reconstruct the existing canonical schema: this also proves that the
    # supplied seen fingerprint/count belong to these very scaler statistics,
    # rather than accepting three individually plausible but unrelated hashes.
    statistics = dict(
        schema=SCHEMA, fit_scope='train_seen_item_rows', accumulation_dtype='float64',
        output_dtype='float32', ddof=0, n_samples_seen=len(seen),
        content_shape=list(content.shape), content_dtype=np.dtype(content._content.dtype).str,
        content_sha256=content.content_sha256, seen_rows_sha256=binding['seen_rows_sha256'],
        mean=content.mean_.tolist(), var=content.var_.tolist(), scale=content.scale_.tolist(),
    )
    if statistics_digest(statistics) != binding['standardization_stats_sha256']:
        raise ValueError('Seen rows and scaler moments do not reproduce the bound statistics')
    return binding


def _hash_chunk(digest, values):
    # Native little-endian machines expose this memory without copying. On
    # another byte order conversion is bounded to this chunk, never full cache.
    canonical = np.asarray(values, dtype='<f4', order='C')
    digest.update(memoryview(canonical).cast('B'))


def _global_indices(selection, size):
    """Mirror StandardizedContent's intentionally row-only indexing contract."""
    if isinstance(selection, slice):
        for bound in (selection.start, selection.stop):
            if bound is not None and (not isinstance(bound, Integral)
                    or isinstance(bound, (bool, np.bool_)) or not 0 <= bound <= size):
                raise ValueError('Slice row bounds must be nonnegative and within the asset')
        if selection.step is not None and not _positive_integer(selection.step):
            raise ValueError('Slice step must be a positive integer')
        start, stop, step = selection.indices(size)
        return np.arange(start, stop, step, dtype=np.int64)
    if isinstance(selection, tuple):
        raise ValueError('Only global row indexing is supported')
    indices = np.asarray(selection)
    if indices.dtype.kind not in 'iu' or (indices < 0).any() or (indices >= size).any():
        raise ValueError('Global rows must be nonnegative in-range integers')
    return indices


class CachedSeenContent:
    """Global-row matrix: train-seen lookups use RAM, cold lookups use base.

    Construct through build_cached_seen. Like the original wrapper, only
    explicitly requested rows are materialized. Output shape still addresses
    the FULL asset, preserving full_catalog_cross_entropy's masked_count.
    """
    ndim = 2
    dtype = np.dtype('float32')

    def __init__(self, content, seen, values, binding, summary):
        if (values.shape != (len(seen), content.shape[1]) or values.dtype != self.dtype
                or not values.flags.c_contiguous or values.flags.writeable or seen.flags.writeable):
            raise ValueError('Cache and row identities must be read-only with the exact seen layout')
        self.base_content, self.seen_rows = content, seen
        self.shape = tuple(content.shape)
        self.stats_sha256, self.content_sha256 = content.stats_sha256, content.content_sha256
        self._values = values
        self._binding = MappingProxyType(dict(binding))
        self._summary = dict(summary)

    def __len__(self):
        return self.shape[0]

    @property
    def cache(self):
        # Read-only view of a read-only owning array: callers cannot accidentally
        # re-enable writes on the public view. There is no second matrix copy.
        view = self._values.view()
        view.flags.writeable = False
        return view

    @property
    def binding(self):
        return self._binding

    @property
    def summary(self):
        return dict(self._summary)

    def _check_source_binding(self):
        if (self.base_content.stats_sha256 != self._binding['standardization_stats_sha256']
                or self.base_content.content_sha256 != self._binding['source_sha256']):
            raise ValueError('Source binding changed after cache construction')

    def __getitem__(self, selection):
        self._check_source_binding()
        indices = _global_indices(selection, len(self))
        flat = indices.reshape(-1)
        shape = indices.shape + (self.shape[1],)
        if not len(flat):
            return np.empty(shape, dtype=np.float32)
        positions = np.searchsorted(self.seen_rows, flat)
        safe = np.minimum(positions, len(self.seen_rows) - 1)
        cached = (positions < len(self.seen_rows)) & (self.seen_rows[safe] == flat)
        if cached.all():
            return self._values[positions].reshape(shape)
        if not cached.any():
            return self.base_content[indices]
        result = np.empty((len(flat), self.shape[1]), dtype=np.float32)
        result[cached] = self._values[positions[cached]]
        result[~cached] = self.base_content[flat[~cached]]
        return result.reshape(shape)


class CachedSeenCatalog:
    """Seen-position catalog; full-loss slices bypass global binary search."""
    def __init__(self, content):
        if not isinstance(content, CachedSeenContent):
            raise ValueError('CachedSeenCatalog requires the matching global-row wrapper')
        self.content, self.rows = content, content.seen_rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, selection):
        self.content._check_source_binding()
        # Validate indexing against one-dimensional catalog rows first, so a
        # tuple cannot accidentally become a selector over feature columns.
        self.rows[selection]
        # An ellipsis in a row selector must never consume the trailing feature
        # dimension: rows[..., None] maps to values[..., None, :], not to
        # values[..., None]. Ordinary loss slices still take the direct path.
        if isinstance(selection, tuple):
            selection = selection + (slice(None),)
        return self.content._values[selection]


def build_cached_seen(content, seen, *, chunk_size=65536, expected_binding):
    """Return (global_wrapper, direct_seen_catalog, JSON-safe build_summary).

    Exactly one len(seen) x feature_dimension FP32 array is allocated. Seen is
    retained by identity and made read-only, not copied. Input source/scaler
    values are unchanged; original asset row zero is a real cached item.
    """
    started = time.monotonic()
    binding = _validate_source(content, seen, expected_binding, chunk_size)
    values = np.empty((len(seen), content.shape[1]), dtype=np.float32)
    digest = hashlib.sha256()
    for start in range(0, len(seen), chunk_size):
        stop = min(start + chunk_size, len(seen))
        transformed = content[seen[start:stop]]
        if transformed.shape != values[start:stop].shape or transformed.dtype != np.float32 or not np.isfinite(transformed).all():
            raise ValueError('Invalid source standardized features')
        values[start:stop] = transformed
        _hash_chunk(digest, values[start:stop])
        del transformed
    values.flags.writeable = False
    seen.flags.writeable = False
    summary = dict(
        schema='liger.train_seen_feature_cache.v1', storage='cpu_ram',
        kind='fixed_standardized_input_features_not_projection',
        rows=len(seen), dimensions=content.shape[1], asset_rows=len(content),
        cache_bytes=values.nbytes, cache_dtype='float32',
        hash_encoding='C-order little-endian float32 bytes', cache_sha256=digest.hexdigest(),
        build_chunk_size=int(chunk_size), build_chunks=int((len(seen) + chunk_size - 1) // chunk_size),
        build_seconds=time.monotonic() - started, **binding)
    wrapper = CachedSeenContent(content, seen, values, binding, summary)
    return wrapper, CachedSeenCatalog(wrapper), dict(summary)


def verify_cached_seen(content, catalog, *, expected_binding, chunk_size=65536):
    """Independently recompute every seen row by chunks and compare FP32 bits."""
    started = time.monotonic()
    if not isinstance(content, CachedSeenContent) or not isinstance(catalog, CachedSeenCatalog):
        raise ValueError('Expected the cached wrapper and direct catalog pair')
    if catalog.content is not content or catalog.rows is not content.seen_rows:
        raise ValueError('Cached wrapper/catalog row identities disagree')
    binding = _validate_source(content.base_content, content.seen_rows, expected_binding, chunk_size)
    if dict(content.binding) != binding:
        raise ValueError('Expected binding differs from the constructed cache')
    values = content._values
    if (values.shape != (len(content.seen_rows), content.shape[1]) or values.dtype != np.float32
            or not values.flags.c_contiguous or values.flags.writeable or catalog.rows.flags.writeable):
        raise ValueError('Cache or seen row storage is no longer immutable/consistent')
    digest = hashlib.sha256()
    for start in range(0, len(catalog), chunk_size):
        stop = min(start + chunk_size, len(catalog))
        expected = content.base_content[catalog.rows[start:stop]]
        actual = values[start:stop]
        if (actual.shape != expected.shape or expected.dtype != np.float32
                or not np.array_equal(actual.view(np.uint32), expected.view(np.uint32))):
            raise ValueError('Cached standardized features differ from an independent source recomputation')
        _hash_chunk(digest, actual)
        del expected
    cache_sha = digest.hexdigest()
    if cache_sha != content.summary['cache_sha256'] or values.nbytes != content.summary['cache_bytes']:
        raise ValueError('Cache content SHA256 or byte count changed')
    return dict(schema='liger.train_seen_feature_cache_verification.v1', status='passed',
                rows=len(catalog), cache_bytes=values.nbytes, cache_sha256=cache_sha,
                feature_values_bitwise_equal=True, verify_chunk_size=int(chunk_size),
                verify_seconds=time.monotonic() - started, **binding)
