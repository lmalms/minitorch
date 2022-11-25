from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from numba import njit, prange

from minitorch.autodiff import Tensor

from .tensor_data import Shape, Storage, Strides, _Index, _Shape, _Strides
from .tensor_ops import MapProto, TensorOps, shape_broadcast


# JIT compilable utils functions
def index_to_position(index: _Index, strides: _Strides) -> int:
    position = 0
    for i, s in zip(index, strides):
        position += i * s

    return int(position)


def to_index(ordinal: int, shape: _Shape, out_index: _Index) -> None:
    shape = shape[::-1]
    for i, dim in enumerate(shape):
        if i == 0:
            div = 1
        else:
            div = np.prod(shape[:i])
        idx = (ordinal // div) % dim
        out_index[(len(out_index) - 1) - i] = idx


def broadcast_index(
    big_index: _Index,
    big_shape: Shape,
    shape: Shape,
    out_index: _Index,
) -> None:
    for i in range(len(shape)):
        # Get the offset (padding)
        # These are the number of dimensions we have to skip
        # to get to the right dimension in the smaller shape
        offset = i + len(big_shape) - len(shape)

        # Get the shape at the offset
        out_index[i] = big_index[offset] if shape[i] != 1 else 0


index_to_position = njit(inline="always")(index_to_position)
to_index = njit(inline="always")(to_index)
broadcast_index = njit(inline="always")(broadcast_index)


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

        # Placeholders to index into
        in_index, out_index = np.zeros(in_shape), np.zeros(out_shape)

        # Conver to arrays
        in_shape, out_shape = np.array(in_shape), np.array(out_shape)

        for out_position in prange(len(out_storage)):

            # Get index corresponding to out_position in out_tensor
            to_index(out_position, out_shape, out_index)

            # Get corresponding index in possibly smaller in_tensor
            broadcast_index(out_index, out_shape, in_shape, in_index)

            # Get corrsponding position in in_tensor
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
        """
        Higher order tensor zip function.
        """
        zip_fn = tensor_zip(njit()(fn))

        def _zip_fn(a: Tensor, b: Tensor) -> Tensor:
            if a.shape != b.shape:
                out_shape = shape_broadcast(a.shape, b.shape)
            else:
                out_shape = a.shape
            out = a.zeros(out_shape)
            zip_fn(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return _zip_fn

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0
    ) -> Callable[[Tensor, int], Tensor]:
        raise NotImplementedError

    @staticmethod
    def matrix_multiply(x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError
