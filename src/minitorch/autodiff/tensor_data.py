from __future__ import annotations

import random
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from numba.cuda import is_cuda_array, to_device
from typing_extensions import TypeAlias

from minitorch.functional import multiply_lists, product, summation

# Types for tensors
_Index: TypeAlias = np.ndarray[np.int32]
_Shape: TypeAlias = np.ndarray[np.int32]
_Strides: TypeAlias = np.ndarray[np.int32]

Storage: TypeAlias = np.ndarray[np.float64]
Index: TypeAlias = Sequence[int]
Shape: TypeAlias = Sequence[int]
Strides: TypeAlias = Sequence[int]


### Utils functions to work with TensorData ###
MAX_DIMS = 32


def index_to_position(index: _Index, strides: _Strides) -> int:
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


def to_index(ordinal: int, shape: _Shape, out_index: _Index) -> None:
    """
    Convert an 'ordinal' to an index in 'shape'.

    Args:
        ordinal : ordinal position to convert
        shape : tensor shape
        out_index : return index corresponding to position
    """

    remaining_ordinal = ordinal
    for i, dim in enumerate(shape):
        is_last_dim = i == (len(shape) - 1)

        if not is_last_dim:
            remaining_size = product(shape[(i + 1) :].tolist())
            idx = remaining_ordinal // remaining_size
            remaining_ordinal = remaining_ordinal % remaining_size
            out_index[i] = idx
        else:
            if remaining_ordinal // shape[i - 1] == 0:
                out_index[i] = remaining_ordinal
            else:
                out_index[i] = remaining_ordinal % dim


def broadcast_index(
    big_index: _Index, big_shape: _Shape, shape: _Shape, out_index: _Index
) -> None:
    for i in range(len(shape)):
        offset = i + len(big_shape) - len(shape)
        out_index[i] = big_index[offset] if shape[i] != 1 else 0


def shape_broadcast(shape_a: Shape, shape_b: Shape) -> Shape:
    """
    Brroadcast two shpes to create a new union shape.
    Args:
        shape_a: first tensor shape
        shape_b: second tensor shape

    Returns:
        broadcasted shape.

    Raises:
        IndexingError: if shapes cannot be broadcast.
    """

    def expand_dims(*dims: Iterable[Shape]):
        max_dim = max(len(dim) for dim in dims)
        dims = [(1,) * (max_dim - len(dim)) + dim for dim in dims]
        return dims

    if len(shape_a) != len(shape_b):
        # Expand dimension to match
        shape_a, shape_b = expand_dims(shape_a, shape_b)

    broadcast_shape = tuple()
    for (a, b) in zip(shape_a, shape_b):
        if a != b:
            if min(a, b) != 1:
                raise IndexError(
                    f"Shapes {shape_a} and {shape_b} cannot be broadcast together."
                )

        broadcast_shape += (max(a, b),)

    return broadcast_shape


def strides_from_shape(shape: Shape) -> Strides:

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


class TensorData:
    _storage: Storage
    _strides: _Strides
    _shape: _Shape
    strides: Strides
    shape: Shape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: Shape,
        strides: Optional[Strides] = None,
    ):
        self._verify_types(shape, strides)
        self._verify_data(storage, strides, shape)
        self._storage = np.array(storage)
        self._shape = np.array(shape)
        self._strides = (
            np.array(strides)
            if strides is not None
            else np.array(strides_from_shape(shape))
        )

    @property
    def storage(self) -> Storage:
        return self._storage

    @property
    def shape(self) -> Shape:
        return tuple(self._shape)

    @property
    def size(self) -> int:
        return int(product(self._shape.tolist()))

    @property
    def dims(self) -> int:
        return len(self.shape)

    @property
    def strides(self) -> Strides:
        return tuple(self._strides)

    @staticmethod
    def _verify_types(shape: Any, strides: Any) -> None:
        if not isinstance(shape, tuple):
            raise TypeError("shape must be a tuple")
        if strides is not None:
            if not isinstance(strides, tuple):
                raise TypeError("strides must be a tuple.")

    @staticmethod
    def _verify_data(storage: Storage, strides: Strides, shape: Shape):
        if strides is None:
            strides = strides_from_shape(shape)

        if len(strides) != len(shape):
            raise IndexError("strides and shape must have the same length.")

        if isinstance(storage, np.ndarray):
            if storage.ndim != 1:
                raise IndexError("Numpy arrays should be 1D.")

        if len(storage) != int(product(list(shape))):
            raise IndexError("data does not match size of tensor.")

    def _verify_index(self, index) -> None:
        if len(index) != len(self.shape):
            raise IndexError(f"Index {index} must be the same size as {self._shape}.")

        for i, idx in enumerate(index):
            if idx >= self.shape[i]:
                raise IndexError(f"Index {idx} of of range for shape {self.shape}.")
            if idx < 0:
                raise IndexError(f"Negative indexing for {index} not supported.")

    def to_cuda_(self) -> None:
        if not is_cuda_array(self._storage):
            self._storage = to_device(self._storage)

    def is_contiguous(self) -> bool:
        """
        Checks that layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool - True if contiguous
        """
        paired_dims = list(zip(self.strides, self.strides[1:]))
        not_contiguous = any(i < j for (i, j) in paired_dims)
        return not not_contiguous

    @staticmethod
    def shape_broadcast(shape_a: Shape, shape_b: Shape) -> Shape:
        return shape_broadcast(shape_a, shape_b)

    def index_to_position(self, index: Union[int, Index]) -> int:
        index = np.array([index]) if isinstance(index, int) else np.array(index)
        self._verify_index(index)
        return index_to_position(np.array(index), self._strides)

    def indices(self) -> Iterable[Index]:
        shape, out_index = np.array(self.shape), np.array(self.shape)
        for i in range(self.size):
            to_index(i, shape, out_index)
            yield tuple(out_index)

    def sample(self) -> Index:
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: Index) -> float:
        return self.storage[self.index_to_position(key)]

    def set(self, key: Index, value: float) -> None:
        self._storage[self.index_to_position(key)] = value

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        return self.storage, self.shape, self.strides

    def permute(self, *order: int) -> TensorData:
        """
        Permute the dimensions of the tensor.

        Args:
            order: List[int] - a permutation of the dimensions

        Returns:
            New TensorData with the same storage and a new dimension order
        """

        if list(sorted(order)) != list(range(len(self.shape))):
            raise IndexError(
                f"Must assign a position to search dimension. Shape {self.shape}; Order {order}"
            )

        return TensorData(
            storage=self._storage,
            shape=tuple([self.shape[i] for i in order]),
            strides=tuple([self.strides[i] for i in order]),
        )

    def to_string(self) -> str:
        tensor_string = ""
        for index in self.indices():
            line = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    # If index[i] == 0 start a new array using [
                    # Have this indented by the position at which the 0 occurs
                    # Hence the \t*i below
                    line = "\n" + "\t" * i + "[" + line
                else:
                    break
            tensor_string += line
            tensor_string += f"{self.get(index):3.2f}"
            line = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    line += "]" if i == self.dims - 1 else "\n" + "\t" * i + "]"
                else:
                    break
            if line:
                tensor_string += line
            else:
                tensor_string += " "
        return tensor_string
