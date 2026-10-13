import numpy as np
import pytest

from scripts.probe_sim_teacher import compare


def test_compare_reports_both_rows_and_values():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[1, 2], [3, 6]])
    assert compare(a, b) == dict(shape=[2, 2], equal_values=3, total_values=4,
                                equal_rows=1, max_abs=2., mean_abs=.5)


def test_compare_rejects_broadcasting():
    with pytest.raises(AssertionError):
        compare(np.zeros((2, 2)), np.zeros((2, 1)))
