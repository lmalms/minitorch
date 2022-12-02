from numba import njit
import pytest
import numpy as np
from minitorch.autodiff.tensor_data import _Index, _Shape


def broadcast_index(
    big_index: _Index,
    big_shape: _Shape,
    shape: _Shape,
    out_index: _Index,
) -> None:
    for i in range(len(shape)):
        # Get the offset (padding)
        # These are the number of dimensions we have to skip
        # to get to the right dimension in the smaller shape
        offset = i + len(big_shape) - len(shape)

        # Get the shape at the offset
        if shape[i] > 1:
            out_index[i] = big_index[offset]
        else:
            out_index[i] = 0


broadcast_index = njit(inline="always")(broadcast_index)


def test_numba():
    arr = np.arange(10)
    arr = greater_than_five(arr)
    print(arr)
