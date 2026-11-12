"""Left insertion positions without promoting a complete fixed row catalog.

NumPy searchsorted can cast its entire haystack to a common dtype when query
IDs use a wider dtype. Here only the requested IDs are cast. Sortedness is the
caller's frozen-catalog contract; rescanning all rows per lookup would itself
introduce the O(catalog size) cost this helper avoids.
"""
import numpy as np


def lookup_sorted_rows(rows, queries):
    """Return integer-exact left insertion positions for query IDs.

    Out-of-range queries retain insertion semantics (below dtype range -> zero;
    above -> len(rows)), including unsigned IDs above int64/int32 maxima.
    Unlike mixed signed/unsigned promotion to floats, large IDs stay distinct.
    No truncation or overflowing cast is permitted. Inputs remain unchanged;
    output is an intp ndarray with the query shape, including a scalar's shape.
    """
    if (not isinstance(rows, np.ndarray) or rows.ndim != 1
            or rows.dtype.kind not in 'iu'):
        raise ValueError('Expected a one-dimensional integer sorted row catalog')
    queries = np.asarray(queries)
    if queries.dtype.kind not in 'iu':
        raise ValueError('Expected integer query IDs')
    flat = queries.reshape(-1)
    positions = np.full(len(flat), len(rows), dtype=np.intp)
    if not len(flat) or not len(rows):
        return positions.reshape(queries.shape)
    row_range, query_range = np.iinfo(rows.dtype), np.iinfo(queries.dtype)
    rmin, rmax = int(row_range.min), int(row_range.max)
    qmin, qmax = int(query_range.min), int(query_range.max)
    if rmin > qmax:
        positions.fill(0)
        return positions.reshape(queries.shape)
    if rmax < qmin:
        return positions.reshape(queries.shape)
    # Bounds are converted only after proving that the query dtype represents
    # them. This avoids signed/unsigned comparisons promoting to float64 at
    # 2**63, where distinct large integer identities can become equal.
    below = flat < queries.dtype.type(rmin) if rmin > qmin else np.zeros(len(flat), dtype=np.bool_)
    above = flat > queries.dtype.type(rmax) if rmax < qmax else np.zeros(len(flat), dtype=np.bool_)
    positions[below] = 0
    represented = ~(below | above)
    if represented.any():
        keys = flat[represented].astype(rows.dtype, copy=False)
        positions[represented] = np.searchsorted(rows, keys, side='left')
    return positions.reshape(queries.shape)
