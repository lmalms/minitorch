from typing import Sequence

import numpy as np
from typing_extensions import TypeAlias

from minitorch.functional import multiply_lists, product, summation

# Types for tensors
Storage: TypeAlias = np.ndarray[np.float64]
OutIndex: TypeAlias = np.ndarray[np.int32]
Index: TypeAlias = np.ndarray[np.int32]
Shape: TypeAlias = np.ndarray[np.int32]
Strides: TypeAlias = np.ndarray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """
    Converts a multidimensional tensor 'index' to a single_dimensional position in storage
    based on strides

    Args:
        index : index tuple of ints
        strides : tensor strides

    Returns:
        Position in storage
    """
    index, strides = index.tolist(), strides.tolist()
    return int(summation(multiply_lists(index, strides)))


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """
    Convert an 'ordinal' to an index in 'shape'.

    Args:
        ordinal : ordinal position to convert
        shape : tensor shape
        out_index : return index corresponding to position
    """

    index = []
    remaining_ordinal = ordinal

    for i, dim in enumerate(shape):
        is_last_dim = i == (len(shape) - 1)

        if not is_last_dim:
            remaining_size = product(shape[(i + 1) :].tolist())
            idx = remaining_ordinal // remaining_size
            remaining_ordinal = remaining_ordinal % remaining_size
            index.append(idx)
        else:
            if remaining_ordinal // shape[i - 1] == 0:
                index.append(remaining_ordinal)
            else:
                index.append(remaining_ordinal % dim)

    return tuple(index)


def broadcast_index() -> None:
    pass


def shape_broadcast() -> UserShape:
    pass


def strides_from_shape(shape: UserShape) -> UserStrides:
    """
    Infers strides from shape. For a given dimension this corresponds to the product of all
    remaining dimensions assuming a contiguous "unrolling" i.e. outer dimensions have digger strides
    than inner dimensions.
    """
    strides, offset = [1], 1
    for s in reversed(shape[1:]):
        strides.append(s * offset)
        offset = s * offset
    return tuple(reversed(strides))

