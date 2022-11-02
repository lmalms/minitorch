from typing import List

import pytest

import minitorch.autodiff.tensor_functions as tf
from minitorch.autodiff import Tensor
from minitorch.operators import is_close
from minitorch.tensor_metrics import accuracy, false_positive_rate, true_positive_rate


@pytest.mark.parametrize(
    ["y_true", "y_hat", "true_acc"],
    [
        (
            tf.ones((4,)),
            tf.ones((4,)),
            tf.ones((1,)),
        ),
        (
            tf.ones((4,)),
            tf.zeros((4,)),
            tf.zeros((1,)),
        ),
        (
            tf.ones((4,)),
            tf.tensor([0.0, 0.0, 1.0, 1.0]),
            tf.tensor([0.5]),
        ),
        (
            tf.ones((4,)),
            tf.tensor([1.0, 0.0, 1.0, 1.0]),
            tf.tensor([0.75]),
        ),
    ],
)
def test_accuracy(y_true: Tensor, y_hat: Tensor, true_acc: Tensor):
    predicted_acc = accuracy(y_true, y_hat)
    assert bool(predicted_acc.is_close(true_acc).item())


@pytest.mark.parametrize(
    ["y_true", "y_hat", "true_tpr"],
    [
        (
            tf.ones((4,)),
            tf.ones((4,)),
            tf.ones((1,)),
        ),
        (
            tf.ones((4,)),
            tf.zeros((4,)),
            tf.zeros((1,)),
        ),
        (
            tf.ones((4,)),
            tf.tensor([0.0, 0.0, 1.0, 1.0]),
            tf.tensor([0.5]),
        ),
        (
            tf.tensor([0.0, 0.0, 1.0, 1.0]),
            tf.tensor([0.0, 1.0, 1.0, 1.0]),
            tf.tensor([1.0]),
        ),
    ],
)
def test_true_positive_rate(y_true: Tensor, y_hat: Tensor, true_tpr: Tensor):
    predicted_tpr = true_positive_rate(y_true, y_hat)
    assert bool(predicted_tpr.is_close(true_tpr).item())


@pytest.mark.parametrize(
    ["y_true", "y_hat", "true_fpr"],
    [
        (
            tf.ones((4,)),
            tf.ones((4,)),
            tf.zeros((1,)),
        ),
        (
            tf.ones((4,)),
            tf.zeros((4,)),
            tf.zeros((1,)),
        ),
        (
            tf.tensor([0.0, 1.0, 0.0, 1.0]),
            tf.tensor([1.0, 0.0, 1.0, 1.0]),
            tf.tensor([1.0]),
        ),
        (
            tf.tensor([0.0, 0.0, 1.0, 0.0]),
            tf.tensor([0.0, 1.0, 1.0, 0.0]),
            tf.tensor([1 / 3]),
        ),
    ],
)
def test_false_positive_rate(y_true: Tensor, y_hat: Tensor, true_fpr: Tensor):
    predicted_fpr = false_positive_rate(y_true, y_hat)
    assert bool(predicted_fpr.is_close(true_fpr).item())
