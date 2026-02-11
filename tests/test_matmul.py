"""Basic unit tests for matmul."""

import numpy as np
import pytest

from mpipy.matmul import mat_mul


def test_matmul_small():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    expected = np.array([[19, 22], [43, 50]])
    result = mat_mul(a, b)
    np.testing.assert_array_equal(result, expected)


def test_matmul_rectangular():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[7, 8], [9, 10], [11, 12]])
    expected = np.array([[58, 64], [139, 154]])
    result = mat_mul(a, b)
    np.testing.assert_array_equal(result, expected)


def test_matmul_rejects_bad_dims():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        mat_mul(a, b)

