import bisect
import tracemalloc

import numpy as np
import pytest

from src.baselines.liger_row_lookup import lookup_sorted_rows


@pytest.mark.parametrize('rows,queries', [
    (np.array([0, 3, 9], dtype=np.int32), np.array([-1, 0, 1, 9, 10, 2**31, 2**32], dtype=np.int64)),
    (np.array([0, 3, 2**31 - 1], dtype=np.int32), np.array([0, 2**31 - 1, 2**31, 2**64 - 1], dtype=np.uint64)),
    (np.array([-2**63, -1, 0, 2**63 - 1], dtype=np.int64), np.array([0, 2**63 - 1, 2**63, 2**64 - 1], dtype=np.uint64)),
    (np.array([0, 2**63 - 1, 2**63, 2**64 - 1], dtype=np.uint64), np.array([-2**63, -1, 0, 2**63 - 1], dtype=np.int64)),
    (np.array([0, 1, 255], dtype=np.uint8), np.array([-10, -1, 0, 127], dtype=np.int8)),
    (np.array([-128, -1, 0, 127], dtype=np.int8), np.array([0, 127, 128, 255], dtype=np.uint8)),
    (np.array([0, 3, 9], dtype=np.int32), np.array([[0, 2], [9, 12]], dtype=np.int64)),
    (np.array([0, 3], dtype=np.int32), np.array(3, dtype=np.uint64)),
    (np.array([], dtype=np.int32), np.array([-1, 2**32], dtype=np.int64)),
    (np.array([0, 3], dtype=np.int32), np.empty((2, 0), dtype=np.int64)),
])
def test_integer_boundaries_shapes_and_left_insertion_match_exact_bisect(rows, queries):
    original_rows, original_queries = rows.copy(), queries.copy()
    expected = np.array([bisect.bisect_left([int(row) for row in rows], int(key))
                         for key in queries.reshape(-1)], dtype=np.intp).reshape(queries.shape)
    actual = lookup_sorted_rows(rows, queries)
    assert actual.dtype == np.intp and actual.shape == queries.shape
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(rows, original_rows)
    np.testing.assert_array_equal(queries, original_queries)


def test_search_uses_original_haystack_and_only_casts_representable_query_subset(monkeypatch):
    rows = np.array([0, 3, 2**31 - 1], dtype=np.int32)
    queries = np.array([0, 2**31 - 1, 2**31, 2**64 - 1], dtype=np.uint64)
    original, calls = np.searchsorted, []
    def checked(haystack, values, side):
        assert haystack is rows and values.dtype == rows.dtype
        calls.append(values.copy())
        return original(haystack, values, side=side)
    monkeypatch.setattr(np, 'searchsorted', checked)
    np.testing.assert_array_equal(lookup_sorted_rows(rows, queries), [0, 2, 3, 3])
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], [0, 2**31 - 1])


def test_queries_outside_catalog_dtype_do_not_invoke_search_or_wrap(monkeypatch):
    monkeypatch.setattr(np, 'searchsorted', lambda *a, **k: pytest.fail('No representable query'))
    rows = np.array([0, 2], dtype=np.int32)
    np.testing.assert_array_equal(lookup_sorted_rows(rows, np.array([2**31, 2**64 - 1], dtype=np.uint64)), [2, 2])
    rows = np.array([0, 2], dtype=np.uint64)
    np.testing.assert_array_equal(lookup_sorted_rows(rows, np.array([-2, -1], dtype=np.int64)), [0, 0])


@pytest.mark.parametrize('rows,queries', [
    ([0, 2], np.array([0])), (np.array([[0, 2]]), np.array([0])),
    (np.array([0., 2.]), np.array([0])), (np.array([0, 2]), np.array([0.])),
    (np.array([0, 2]), np.array([True])), (np.array([False, True]), np.array([0])),
])
def test_invalid_catalog_or_query_types_rejected(rows, queries):
    with pytest.raises(ValueError): lookup_sorted_rows(rows, queries)


def test_short_query_does_not_allocate_an_int64_copy_of_large_int32_catalog():
    if tracemalloc.is_tracing(): pytest.skip('Do not disturb an external memory profiler')
    rows = np.arange(262144, dtype=np.int32)
    keys = np.array([0, 1, 65536, 262143], dtype=np.int64)
    tracemalloc.start()
    try:
        actual = lookup_sorted_rows(rows, keys)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    np.testing.assert_array_equal(actual, keys)
    # The old mixed-dtype call allocates about2MiB. Allow64KiB for tiny query
    # masks/results and interpreter variation, without a noisy time threshold.
    assert peak < 65536, f'Unexpected catalog-sized temporary allocation: {peak}'
