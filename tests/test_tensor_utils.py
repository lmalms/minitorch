import pytest

import minitorch.autodiff.tensor_functions as tf
from minitorch.autodiff import Tensor
from minitorch.tensor_losses import _check_dims
from minitorch.tensor_metrics import _check_for_binary_values


@pytest.mark.parametrize(
    ["x", "y"],
    [
        (tf.ones((5, 1)), tf.ones((5, 1))),
        (tf.ones((1, 1)), tf.ones((1, 1))),
    ],
)
def test_equal_dims_check(x: Tensor, y: Tensor) -> None:
    _check_dims(x, y)


@pytest.mark.xfail
@pytest.mark.parametrize(
    ["x", "y"],
    [
        (tf.ones((5, 1)), tf.ones((4, 1))),
        (tf.ones((1,)), tf.ones((1,))),
        (tf.ones((1, 2)), tf.ones((1, 2))),
    ],
)
def test_equal_dims_check_fail(x: Tensor, y: Tensor) -> None:
    _check_dims(x, y)


@pytest.mark.parametrize(
    "y",
    [tf.ones((5,)), tf.zeros((5,)), tf.tensor([1.0, 0.0, 1.0, 1.0, 0.0])],
)
def test_binary_values_check(y: Tensor) -> None:
    _check_for_binary_values(y)


@pytest.mark.xfail
@pytest.mark.parametrize(
    "y",
    [
        tf.tensor([1.1, 0.0, 1.0, 1.0, 0.0]),
        tf.tensor([1.0, 0.5, 1.0, 1.0, 0.0]),
        tf.tensor([1.0, 0.3, 1.0, 1.3, 0.0]),
    ],
)
def test_binary_values_check_fail(y: Tensor) -> None:
    _check_for_binary_values(y)
