import pytest
from hypothesis import given
from hypothesis.strategies import DataObject, data

import numpy as np
from minitorch.autodiff import TensorData
from .tensor_strategies import tensor_data
from minitorch.functional import product


def test_tensor_data_layout() -> None:
    """
    Test basic properties of layout and strides.
    """

    shape, strides = (3, 5), (5, 1)
    size = int(product(list(shape)))
    td = TensorData([0] * size, shape, strides)
    assert td.is_contiguous()
    assert np.all(td.shape == np.array(shape))
    assert td.index((1, 0)) == 5
    assert td.index((1, 2)) == 7

    shape, strides = (5, 3), (1, 5)
    size = int(product(list(shape)))
    td = TensorData([0] * size, shape, strides)
    assert np.all(td.shape == np.array(shape))
    assert not td.is_contiguous()

    shape = (4, 2, 2)
    size = int(product(list(shape)))
    td = TensorData([0] * size, shape)
    assert np.all(td.strides == np.array((4, 2, 1)))


@pytest.mark.xfail
def test_tensor_data_layout_fail() -> None:
    """
    Make sure that bad layouts fail.
    """
    shape, strides = (3, 5), (6, )
    size = int(product(list(shape)))
    TensorData([0] * size, shape, strides)
