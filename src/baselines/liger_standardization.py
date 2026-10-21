"""Train-seen-only content standardization for a new LIGER adaptation.

The fitter never scans the held-out/cold complement. It uses float64 centered
chunk moments and Chan's merge, with population variance (ddof=0), matching
StandardScaler. The caller owns persistence and must verify any supplied asset
SHA256 against the immutable source file; this module does not read that file.

One lazy wrapper is shared by history packing, training RowCatalog and cold
retrieval. It transforms actual asset rows, including row zero, never padding:
pack_sequences initializes padding separately and leaves those zeros intact.
"""

import hashlib
import json
import numbers
import re

import numpy as np


SCHEMA = 'liger.train_seen_standardization.v1'
_FIELDS = frozenset({
    'schema', 'fit_scope', 'accumulation_dtype', 'output_dtype', 'ddof',
    'n_samples_seen', 'content_shape', 'content_dtype', 'content_sha256',
    'seen_rows_sha256', 'mean', 'var', 'scale', 'stats_sha256',
})


def _sha256(value, name, *, optional=False):
    if optional and value is None:
        return
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError(f'{name} must be a lowercase SHA256 digest')


def _content_matrix(content):
    if (getattr(content, 'ndim', None) != 2 or len(content.shape) != 2
            or min(content.shape) < 1
            or np.dtype(content.dtype).kind not in 'iuf'):
        raise ValueError('Expected a nonempty real numeric content matrix')


def _positive_integer(value):
    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_)) and value > 0


def _scale_from_moments(mean, var, count):
    # StandardScaler treats numerically indistinguishable constant features as
    # constant too. This bound avoids amplifying roundoff around large means.
    eps = np.finfo(np.float64).eps
    with np.errstate(over='ignore', invalid='ignore'):
        upper_bound = count * eps * var + (count * mean * eps) ** 2
    scale = np.sqrt(var)
    scale[var <= upper_bound] = 1.
    return scale


def statistics_digest(statistics):
    """Canonical JSON SHA256 of all fields except the digest itself."""
    payload = {key: value for key, value in statistics.items() if key != 'stats_sha256'}
    try:
        encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'),
                             ensure_ascii=True, allow_nan=False).encode('ascii')
    except (ValueError, TypeError) as exc:
        raise ValueError('Statistics must be finite JSON values') from exc
    return hashlib.sha256(encoded).hexdigest()


def fit_standardization(content, seen_rows, *, chunk_size=65536, content_sha256=None):
    """Return a JSON-serializable, hash-bound train-seen StandardScaler state.

    ``seen_rows`` is the fixed sorted unique training-consumed item union, not
    validation/test targets or the full asset catalog. Row IDs are hashed as
    ordered little-endian int64 bytes, independent of the source integer dtype.
    Each selected item contributes once. ``content_sha256`` is optional for
    in-memory fixtures; production manifests should bind the verified asset.
    """
    _content_matrix(content)
    _sha256(content_sha256, 'content_sha256', optional=True)
    rows = np.asarray(seen_rows)
    if (rows.ndim != 1 or not len(rows) or rows.dtype.kind not in 'iu'
            or (rows < 0).any() or (rows >= len(content)).any()
            or not (rows[1:] > rows[:-1]).all()):
        raise ValueError('Expected nonempty sorted unique train-seen rows within the asset')
    if not _positive_integer(chunk_size):
        raise ValueError('chunk_size must be a positive integer')

    dimension, count = content.shape[1], 0
    mean, m2 = np.zeros(dimension, dtype=np.float64), np.zeros(dimension, dtype=np.float64)
    rows_digest = hashlib.sha256()
    for start in range(0, len(rows), chunk_size):
        selected = rows[start:start + chunk_size]
        rows_digest.update(np.asarray(selected, dtype='<i8').tobytes(order='C'))
        values = np.asarray(content[selected], dtype=np.float64)
        if values.shape != (len(selected), dimension) or not np.isfinite(values).all():
            raise ValueError('Nonfinite or invalid train-seen content')
        with np.errstate(over='ignore', invalid='ignore'):
            local_mean = values.mean(axis=0, dtype=np.float64)
            centered = values - local_mean
            local_m2 = np.einsum('ij,ij->j', centered, centered)
            combined_count = count + len(values)
            delta = local_mean - mean
            m2 += local_m2 + delta * delta * (count * len(values) / combined_count)
            mean += delta * (len(values) / combined_count)
        count = combined_count
        if not np.isfinite(mean).all() or not np.isfinite(m2).all():
            raise ValueError('Train-seen moments overflowed float64')
    var = m2 / count
    statistics = dict(
        schema=SCHEMA, fit_scope='train_seen_item_rows', accumulation_dtype='float64',
        output_dtype='float32', ddof=0, n_samples_seen=count,
        content_shape=list(content.shape), content_dtype=np.dtype(content.dtype).str,
        content_sha256=content_sha256, seen_rows_sha256=rows_digest.hexdigest(),
        mean=mean.tolist(), var=var.tolist(), scale=_scale_from_moments(mean, var, count).tolist(),
    )
    statistics['stats_sha256'] = statistics_digest(statistics)
    validate_standardization(statistics, content=content, expected_content_sha256=content_sha256)
    return statistics


def validate_standardization(statistics, *, content=None, expected_content_sha256=None):
    """Reject malformed, unbound/tampered or incompatible statistics.

    A matching supplied source SHA is an identity assertion from the caller,
    not a new hash of the whole source matrix. Return independent float64
    mean/variance/scale arrays so a wrapper cannot be changed by mutating JSON.
    """
    if not isinstance(statistics, dict) or set(statistics) != _FIELDS:
        raise ValueError('Invalid standardization schema fields')
    if (statistics['schema'] != SCHEMA or statistics['fit_scope'] != 'train_seen_item_rows'
            or statistics['accumulation_dtype'] != 'float64'
            or statistics['output_dtype'] != 'float32'
            or type(statistics['ddof']) is not int or statistics['ddof'] != 0):
        raise ValueError('Unsupported standardization schema or policy')
    shape, count = statistics['content_shape'], statistics['n_samples_seen']
    if (not isinstance(shape, list) or len(shape) != 2 or not all(_positive_integer(x) for x in shape)
            or not _positive_integer(count) or count > shape[0]):
        raise ValueError('Invalid standardization dimensions or sample count')
    try:
        dtype = np.dtype(statistics['content_dtype'])
    except (ValueError, TypeError) as exc:
        raise ValueError('Invalid content dtype') from exc
    if dtype.kind not in 'iuf' or statistics['content_dtype'] != dtype.str:
        raise ValueError('Content dtype must use canonical real numeric notation')
    _sha256(statistics['content_sha256'], 'content_sha256', optional=True)
    _sha256(statistics['seen_rows_sha256'], 'seen_rows_sha256')
    _sha256(statistics['stats_sha256'], 'stats_sha256')
    _sha256(expected_content_sha256, 'expected_content_sha256', optional=True)
    if expected_content_sha256 is not None and statistics['content_sha256'] != expected_content_sha256:
        raise ValueError('Source content SHA256 does not match the bound statistics')
    vectors = []
    for name in ('mean', 'var', 'scale'):
        value = statistics[name]
        if (not isinstance(value, list) or len(value) != shape[1]
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value)):
            raise ValueError(f'Invalid {name} vector')
        vector = np.asarray(value, dtype=np.float64)
        if not np.isfinite(vector).all():
            raise ValueError(f'Nonfinite {name} vector')
        vectors.append(vector)
    mean, var, scale = vectors
    if (var < 0).any() or (scale <= 0).any():
        raise ValueError('Variance must be nonnegative and scale positive')
    if not np.array_equal(scale, _scale_from_moments(mean, var, count)):
        raise ValueError('Scale does not match the population variance and constant-feature rule')
    if statistics['stats_sha256'] != statistics_digest(statistics):
        raise ValueError('Standardization statistics SHA256 mismatch')
    if content is not None:
        _content_matrix(content)
        if tuple(content.shape) != tuple(shape) or np.dtype(content.dtype) != dtype:
            raise ValueError('Source content shape or dtype does not match the statistics')
    return mean, var, scale


class StandardizedContent:
    """Lazy, row-addressed FP32 matrix over an immutable raw content asset.

    Supports nonnegative integer row IDs, integer row arrays, and positive-step
    slices with in-range nonnegative bounds. Negative IDs, boolean masks and
    tuple/column selectors are intentionally rejected to catch identity bugs.
    Only requested rows are materialized; no all-catalog copy or pad transform.
    """

    ndim = 2
    dtype = np.dtype('float32')

    def __init__(self, content, statistics, *, expected_content_sha256=None):
        self.mean_, self.var_, self.scale_ = validate_standardization(
            statistics, content=content, expected_content_sha256=expected_content_sha256)
        for vector in (self.mean_, self.var_, self.scale_):
            vector.flags.writeable = False
        self._content = content
        self.shape = tuple(content.shape)
        self.stats_sha256 = statistics['stats_sha256']
        self.content_sha256 = statistics['content_sha256']

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, selection):
        if isinstance(selection, slice):
            for bound in (selection.start, selection.stop):
                if bound is not None and (not isinstance(bound, numbers.Integral)
                        or isinstance(bound, (bool, np.bool_)) or not 0 <= bound <= len(self)):
                    raise ValueError('Slice row bounds must be nonnegative and within the asset')
            if selection.step is not None and not _positive_integer(selection.step):
                raise ValueError('Slice step must be a positive integer')
        else:
            if isinstance(selection, tuple):
                raise ValueError('Only row indexing is supported')
            indices = np.asarray(selection)
            if indices.dtype.kind not in 'iu' or (indices < 0).any() or (indices >= len(self)).any():
                raise ValueError('Row identities must be nonnegative in-range integers')
            selection = indices
        values = np.asarray(self._content[selection], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite source content in selected rows')
        with np.errstate(over='ignore', invalid='ignore'):
            transformed = ((values - self.mean_) / self.scale_).astype(np.float32)
        if not np.isfinite(transformed).all():
            raise ValueError('Standardized content is outside finite float32 range')
        return transformed
