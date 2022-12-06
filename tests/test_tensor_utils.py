from typing import List

import pytest

import minitorch.autodiff.tensor_functions as tf
from minitorch.autodiff import FastOps, SimpleOps, Tensor, TensorBackend
from minitorch.autodiff.tensor_data import Shape
from minitorch.tensor_losses import _check_dims
from minitorch.tensor_metrics import _check_for_binary_values

# Define backends
BACKENDS = {"simple": TensorBackend(SimpleOps), "fast": TensorBackend(FastOps)}


@pytest.mark.parametrize(
    ["x_shape", "y_shape"],
    [((5, 1), (5, 1)), ((1, 1), (1, 1))],
)
@pytest.mark.parametrize(
    "backend",
    (pytest.param("simple"), pytest.param("fast")),
)
def test_equal_dims_check(x_shape: Shape, y_shape: Shape, backend: str) -> None:
    x = tf.ones(x_shape, backend=BACKENDS[backend])
    y = tf.ones(y_shape, backend=BACKENDS[backend])
    _check_dims(x, y)


@pytest.mark.xfail
@pytest.mark.parametrize(
    ["x_shape", "y_shape"],
    [
        ((5, 1), (4, 1)),
        ((1,), (1,)),
        ((1, 2), (1, 2)),
    ],
)
@pytest.mark.parametrize(
    "backend",
    (pytest.param("simple"), pytest.param("fast")),
)
def test_equal_dims_check_fail(x_shape: Shape, y_shape: Shape, backend: str) -> None:
    x = tf.ones(x_shape, backend=BACKENDS[backend])
    y = tf.ones(y_shape, backend=BACKENDS[backend])
    _check_dims(x, y)


@pytest.mark.parametrize(
    "y",
    [
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 1.0, 0.0],
    ],
)
@pytest.mark.parametrize(
    "backend",
    (pytest.param("simple"), pytest.param("fast")),
)
def test_binary_values_check(y: List[float], backend: str) -> None:
    y = tf.tensor(y, backend=BACKENDS[backend])
    _check_for_binary_values(y)


@pytest.mark.xfail
@pytest.mark.parametrize(
    "y",
    [
        [1.1, 0.0, 1.0, 1.0, 0.0],
        [1.0, 0.5, 1.0, 1.0, 0.0],
        [1.0, 0.3, 1.0, 1.3, 0.0],
    ],
)
@pytest.mark.parametrize(
    "backend",
    (pytest.param("simple"), pytest.param("fast")),
)
def test_binary_values_check_fail(y: List[float], backend: str) -> None:
    y = tf.tensor(y, backend=BACKENDS[backend])
    _check_for_binary_values(y)
