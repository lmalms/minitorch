from __future__ import annotations

from typing import Callable, Optional, Tuple, Iterable

import numpy as np
from numba import njit, jit, prange
from minitorch.functional import product, reduce

from minitorch.autodiff import Tensor

from .tensor_data import (
    MAX_DIMS,
    _Index,
    _Shape,
    Storage,
    _Strides,
    broadcast_index,
    shape_broadcast,
)
from .tensor_ops import MapProto, TensorOps, to_numpy

# JIT compiled functions
# These are a slight re-write of functions in tensor_data to make
# them JIT compatible


def index_to_position(index: _Index, strides: _Strides) -> int:
    position = 0
    for i, s in zip(index, strides):
        position += i * s

    return position


def to_index(ordinal: int, shape: _Shape, out_index: _Index) -> None:
    remaining_ordinal = ordinal
    for i, dim in enumerate(shape[::-1]):
        is_last_dim = i == (len(shape) - 1)
        if is_last_dim:
            out_index[i] = remaining_ordinal
        else:
            out_index[i] = int(remaining_ordinal // dim)
            remaining_ordinal = remaining_ordinal % dim


def broadcast_index(
    big_index: _Index, big_shape: _Shape, shape: _Shape, out_index: _Index
) -> None:
    for i in range(len(shape)):
        offset = i + len(big_shape) - len(shape)
        out_index[i] = big_index[offset] if shape[i] != 1 else 0


# index_to_position = njit(inline="always")(index_to_position)
# broadcast_index = njit(inline="always")(broadcast_index)
# shape_broadcast = njit(inline="always")(shape_broadcast)

# def to_numpy(*inputs: Tuple[float]) -> Iterable[np.ndarray]:
#     return [np.array(in_) for in_ in inputs]


def tensor_map(fn: Callable[[float], float]):
    """
    Low-level implementation of tensor map between
    tensors with possibly different strides.
    """

    def _map(
        out_storage: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:

        # Cast as numpy arrays
        out_shape, out_strides = np.array(out_shape), np.array(out_strides)
        in_shape, in_strides = np.array(in_shape), np.array(in_strides)

        # Placeholders to use during map
        out_size = int(np.prod(out_shape))
        in_index = np.zeros_like(in_shape)
        out_index = np.zeros_like(out_shape)

        for out_position in prange(out_size):
            # Get index corresponding to ordinal in out_tensor
            to_index(out_position, out_shape, out_index)

            # Get corresponding index in possibly smaller in_tensor
            broadcast_index(out_index, out_shape, in_shape, in_index)

            # Get corrsponding ordinal into smaller in_tensor
            in_position = index_to_position(in_index, in_strides)

            # Apply func at positions
            out_storage[out_position] = fn(in_storage[in_position])

    return njit(parallel=True)(_map)


def tensor_zip(fn: Callable[[float, float], float]):
    raise NotImplementedError


def tensor_reduce(fn: Callable[[float, float], float]):
    raise NotImplementedError


def tensor_matrix_multiply():
    raise NotImplementedError


class FastTensorOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """
        Higher order JIT compiled tensor map function
        """
        map_fn = tensor_map(njit()(fn))

        def _map_fn(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            map_fn(*out.tuple(), *a.tuple())
            return out

        return _map_fn

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        raise NotImplementedError

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0
    ) -> Callable[[Tensor, int], Tensor]:
        raise NotImplementedError

    @staticmethod
    def matrix_multiply(x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError
