import pytest

import minitorch.autodiff.tensor_functions as tf
from minitorch.autodiff import Tensor
from minitorch.tensor_losses import _check_dims


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
