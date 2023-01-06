from typing import List

import pytest

from minitorch.autodiff import Scalar
from minitorch.operators import is_close
from minitorch.scalar_metrics import accuracy, false_positive_rate, true_positive_rate


@pytest.mark.parametrize(
    ["y_true", "y_hat", "true_acc"],
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
def test_accuracy(y_true: List[Scalar], y_hat: List[Scalar], true_acc: Scalar):
    predicted_acc = accuracy(y_true, y_hat)
    assert is_close(predicted_acc.data, true_acc.data)


@pytest.mark.parametrize(
    ["y_true", "y_hat", "true_tpr"],
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
            [Scalar(0), Scalar(0), Scalar(1), Scalar(1)],
            [Scalar(0), Scalar(1), Scalar(1), Scalar(1)],
            Scalar(1),
        ),
    ],
)
def test_true_positive_rate(
    y_true: List[Scalar], y_hat: List[Scalar], true_tpr: Scalar
):
    predicted_tpr = true_positive_rate(y_true, y_hat)
    assert is_close(predicted_tpr.data, true_tpr.data)


@pytest.mark.parametrize(
    ["y_true", "y_hat", "true_fpr"],
    [
        (
            [Scalar(1), Scalar(1), Scalar(1), Scalar(1)],
            [Scalar(1), Scalar(1), Scalar(1), Scalar(1)],
            Scalar(0.0),
        ),
        (
            [Scalar(1), Scalar(1), Scalar(1), Scalar(1)],
            [Scalar(0), Scalar(0), Scalar(0), Scalar(0)],
            Scalar(0.0),
        ),
        (
            [Scalar(0), Scalar(1), Scalar(0), Scalar(1)],
            [Scalar(1), Scalar(0), Scalar(1), Scalar(1)],
            Scalar(1.0),
        ),
        (
            [Scalar(0), Scalar(0), Scalar(1), Scalar(0)],
            [Scalar(0), Scalar(1), Scalar(1), Scalar(0)],
            Scalar(value=1 / 3),
        ),
    ],
)
def test_false_positive_rate(
    y_true: List[Scalar], y_hat: List[Scalar], true_fpr: Scalar
):
    predicted_fpr = false_positive_rate(y_true, y_hat)
    assert is_close(predicted_fpr.data, true_fpr.data)
