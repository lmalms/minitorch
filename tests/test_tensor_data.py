import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import DataObject, data

from minitorch.autodiff import Shape, TensorData, shape_broadcast
from minitorch.functional import product

from .tensor_strategies import indices, tensor_data


def test_layout() -> None:
    """
    Test basic properties of layout and strides.
    """

    shape, strides = (3, 5), (5, 1)
    size = int(product(list(shape)))
    td = TensorData([0] * size, shape, strides)
    assert td.is_contiguous()
    assert np.all(td.shape == np.array(shape))
    assert td.index_to_position((1, 0)) == 5
    assert td.index_to_position((1, 2)) == 7

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
def test_layout_fail() -> None:
    """
    Make sure that bad layouts fail.
    """
    shape, strides = (3, 5), (6,)
    size = int(product(list(shape)))
    TensorData([0] * size, shape, strides)


@given(tensor_data())
def test_enumeration(td: TensorData) -> None:
    """Test enumeration of tensors."""
    indices = list(td.indices())

    # Check that every position is enumerated
    assert len(indices) == td.size

    # Check that every position is enumerated only once
    assert len(set(indices)) == len(indices)

    # Check that all indices are within shape
    for index in td.indices():
        for (dim, i) in enumerate(index):
            assert 0 <= i < td.shape[dim]


@given(tensor_data())
def test_index(td: TensorData) -> None:
    """Test enumeration of TensorData."""
    # Check that all indices are within the size.
    for index in td.indices():
        position = td.index_to_position(index)
        assert 0 <= position < td.size

    # Check that negative index raises error
    index = [0] * td.dims
    with pytest.raises(IndexError):
        index[0] = -1
        td.index_to_position(tuple(index))

    if td.dims > 1:
        index = [0] * (td.dims - 1)
        with pytest.raises(IndexError):
            td.index_to_position(tuple(index))


@given(data())
def test_permute(data: DataObject) -> None:
    """Test shape permutation."""
    td = data.draw(tensor_data())
    idx = data.draw(indices(td))
    td_reverse = td.permute(*list(reversed(range(td.dims))))
    assert td.index_to_position(idx) == td_reverse.index_to_position(
        tuple(reversed(idx))
    )

    td_reverse_twice = td_reverse.permute(*list(reversed(range(td_reverse.dims))))
    assert td.index_to_position(idx) == td_reverse_twice.index_to_position(idx)


@given(tensor_data())
def test_string(td: TensorData) -> None:
    """Test tensor data stringify."""
    td.to_string()


@pytest.mark.parametrize(
    "shape_a, shape_b, expected_shape",
    [
        ((1,), (5, 5), (5, 5)),
        ((5, 5), (1,), (5, 5)),
        ((2, 5), (5,), (2, 5)),
        ((1, 5, 5), (5, 5), (1, 5, 5)),
        ((5, 1, 5, 1), (1, 5, 1, 5), (5, 5, 5, 5)),
    ],
)
def test_shape_broadcast(shape_a: Shape, shape_b: Shape, expected_shape: Shape) -> None:
    broadcast_shape = shape_broadcast(shape_a, shape_b)
    assert broadcast_shape == expected_shape


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((5, 2), (5,)),
        ((5, 7, 5, 1), (1, 5, 1, 5)),
    ],
)
def test_shape_broadcast_fail(shape_a: Shape, shape_b: Shape) -> None:
    with pytest.raises(IndexError):
        _ = shape_broadcast(shape_a, shape_b)
