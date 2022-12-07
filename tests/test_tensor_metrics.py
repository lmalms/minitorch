from typing import List

import pytest

import minitorch.autodiff.tensor_functions as tf
from minitorch.autodiff import FastOps, SimpleOps, Tensor, TensorBackend
from minitorch.tensor_metrics import accuracy, false_positive_rate, true_positive_rate

# Define tensor backends
BACKENDS = {"simple": TensorBackend(SimpleOps), "fast": TensorBackend(FastOps)}


@pytest.mark.parametrize(
    ["y_true", "y_hat", "expected_acc"],
    [
        (
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0],
        ),
        ([1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0]),
        (
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.5],
        ),
        (
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [0.75],
        ),
    ],
)
@pytest.mark.parametrize(
    "backend",
    (
        pytest.param("simple"),
        pytest.param("fast"),
    ),
)
def test_accuracy(
    y_true: List[float],
    y_hat: List[float],
    expected_acc: List[float],
    backend: str,
):
    y_true = tf.tensor(y_true, backend=BACKENDS[backend])
    y_hat = tf.tensor(y_hat, backend=BACKENDS[backend])
    expected_acc = tf.tensor(expected_acc, backend=BACKENDS[backend])
    predicted_acc = accuracy(y_true, y_hat)
    assert bool(predicted_acc.is_close(expected_acc).item())


@pytest.mark.parametrize(
    ["y_true", "y_hat", "expected_tpr"],
    [
        (
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0],
        ),
        (
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0],
        ),
        (
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.5],
        ),
        (
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [1.0],
        ),
    ],
)
@pytest.mark.parametrize(
    "backend",
    (
        pytest.param("simple"),
        pytest.param("fast"),
    ),
)
def test_true_positive_rate(
    y_true: List[float],
    y_hat: List[float],
    expected_tpr: List[float],
    backend: str,
):
    y_true = tf.tensor(y_true, backend=BACKENDS[backend])
    y_hat = tf.tensor(y_hat, backend=BACKENDS[backend])
    expected_tpr = tf.tensor(expected_tpr, backend=BACKENDS[backend])
    predicted_tpr = true_positive_rate(y_true, y_hat)
    assert bool(predicted_tpr.is_close(expected_tpr).item())


@pytest.mark.parametrize(
    ["y_true", "y_hat", "expected_fpr"],
    [
        (
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0],
        ),
        (
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0],
        ),
        (
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [1.0],
        ),
        (
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [1 / 3],
        ),
    ],
)
@pytest.mark.parametrize(
    "backend",
    (
        pytest.param("simple"),
        pytest.param("fast"),
    ),
)
def test_false_positive_rate(
    y_true: List[float],
    y_hat: List[float],
    expected_fpr: List[float],
    backend: str,
):
    y_true = tf.tensor(y_true, backend=BACKENDS[backend])
    y_hat = tf.tensor(y_hat, backend=BACKENDS[backend])
    expected_fpr = tf.tensor(expected_fpr, backend=BACKENDS[backend])
    predicted_fpr = false_positive_rate(y_true, y_hat)
    assert bool(predicted_fpr.is_close(expected_fpr).item())
