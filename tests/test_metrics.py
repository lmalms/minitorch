from typing import List

import pytest

from minitorch.autodiff import Scalar
from minitorch.metrics import accuracy
from minitorch.operators import is_close


@pytest.mark.parametrize(
    ["y_true", "y_hat", "expected"],
    [
        (
            [Scalar(1), Scalar(1), Scalar(1), Scalar(1)],
            [Scalar(1), Scalar(1), Scalar(1), Scalar(1)],
            Scalar(1),
        ),
        (
            [Scalar(1), Scalar(1), Scalar(1), Scalar(1)],
            [Scalar(0), Scalar(0), Scalar(0), Scalar(0)],
            Scalar(0),
        ),
        (
            [Scalar(1), Scalar(1), Scalar(1), Scalar(1)],
            [Scalar(0), Scalar(0), Scalar(1), Scalar(1)],
            Scalar(0.5),
        ),
        (
            [Scalar(1), Scalar(1), Scalar(1), Scalar(1)],
            [Scalar(0), Scalar(1), Scalar(1), Scalar(1)],
            Scalar(0.75),
        ),
    ],
)
def test_accuracy(y_true: List[Scalar], y_hat: List[Scalar], expected: Scalar):
    acc = accuracy(y_true, y_hat)
    assert is_close(acc.data, expected.data)
