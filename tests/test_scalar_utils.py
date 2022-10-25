from typing import List

import pytest

from minitorch.autodiff import Scalar
from minitorch.scalar_metrics import _check_for_binary_values, _check_for_equal_dim


@pytest.mark.parametrize(
    "y",
    [
        [Scalar(1), Scalar(1), Scalar(1)],
        [Scalar(0), Scalar(0), Scalar(0)],
        [Scalar(1), Scalar(0), Scalar(0)],
    ],
)
def test_binary_values_check(y: List[Scalar]) -> None:
    _check_for_binary_values(y)


@pytest.mark.xfail
@pytest.mark.parametrize(
    "y",
    [
        [Scalar(1.1), Scalar(1), Scalar(1)],
        [Scalar(0), Scalar(0.5), Scalar(0)],
        [Scalar(1), Scalar(0.3), Scalar(0)],
    ],
)
def test_binary_values_check_fail(y: List[Scalar]) -> None:
    _check_for_binary_values(y)


@pytest.mark.parametrize(
    ["x", "y"],
    [
        (
            [Scalar(1), Scalar(1)],
            [Scalar(0), Scalar(0)],
        ),
        (
            [Scalar(1)],
            [Scalar(0)],
        ),
    ],
)
def test_equal_dims_check(x: List[Scalar], y: List[Scalar]) -> None:
    _check_for_equal_dim(x, y)


@pytest.mark.xfail
@pytest.mark.parametrize(
    ["x", "y"],
    [
        (
            [Scalar(1)],
            [Scalar(0), Scalar(0)],
        ),
        (
            [Scalar(1)],
            [Scalar(0), Scalar(1)],
        ),
    ],
)
def test_equal_dims_check_fail(x: List[Scalar], y: List[Scalar]) -> None:
    _check_for_equal_dim(x, y)
