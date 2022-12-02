from __future__ import annotations

from functools import reduce
from typing import Callable, Optional

import numpy as np
from numba import njit, prange

import minitorch.autodiff.tensor as t

from .tensor_data import Shape, Storage, Strides, _Index, _Shape, _Strides
from .tensor_ops import MapProto, TensorOps, shape_broadcast


# JIT compilable utils functions
def index_to_position(index: _Index, strides: _Strides) -> int:
    prod = np.multiply(index, strides)
    return int(np.sum(prod))


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
        out_index[i] = big_index[offset] if shape[i] > 1 else 0


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
        out_shape, out_index = np.array(out_shape), np.zeros_like(
            out_shape, dtype=np.int64
        )
        in_shape, in_index = np.array(in_shape), np.zeros_like(in_shape, dtype=np.int64)
        in_strides = np.array(in_strides)

        print(out_index)
        print(in_index)

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
    """
    Low-level implementation of tensor map between
    tensors with possibly different strides.
    """

    def _zip(
        out_storage: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # Placeholders to index into
        out_shape, out_index = np.array(out_shape), np.zeros_like(out_shape)
        a_shape, a_index = np.array(a_shape), np.zeros_like(a_shape)
        b_shape, b_index = np.array(b_shape), np.zeros_like(b_shape)
        a_strides, b_strides = np.array(a_strides), np.array(b_strides)

        for out_position in prange(len(out_storage)):

            # Grab the index in out from position
            to_index(out_position, out_shape, out_index)

            # Get the corresponding positions in possibly smaller in_tensors
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            # From these indices get positions
            a_position = index_to_position(a_index, a_strides)
            b_position = index_to_position(b_index, b_strides)

            # Apply func at position
            out_storage[out_position] = fn(a_storage[a_position], b_storage[b_position])

    return njit(parallel=True)(_zip)


def tensor_reduce(fn: Callable[[float, float], float]):
    """
    Low-level implementation of tensor map between
    tensors with possibly different strides.
    """

    def _reduce(
        out_storage: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
        reduce_dim: int,
    ):
        # Cast to numpy arrays
        out_shape, out_index = np.array(out_shape), np.zeros(out_shape, dtype=np.int64)
        in_shape, in_index = np.array(in_shape), np.zeros(in_shape, dtype=np.int64)
        in_strides = np.array(in_strides)

        for out_position in prange(len(out_storage)):
            # Grab the corresponding out_index
            to_index(out_position, out_shape, out_index)

            # Get all positions that will be reduced to that out_index
            in_values = []
            in_index = out_index[:]
            for j in prange(in_shape[reduce_dim]):
                in_index[reduce_dim] = j
                in_position = index_to_position(in_index, in_strides)
                in_values.append(in_storage[in_position])

            # Get all of the corresponding values
            out_value = reduce(fn, np.array(in_values), out_storage[out_position])
            out_storage[out_position] = out_value

    return njit(parallel=True)(_reduce)


def tensor_matrix_multiply():
    raise NotImplementedError


class FastTensorOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """
        Higher order JIT compiled tensor map function
        """
        map_fn = tensor_map(njit()(fn))

        def _map_fn(a: t.Tensor, out: Optional[t.Tensor] = None) -> t.Tensor:
            if out is None:
                out = a.zeros(a.shape)
            map_fn(*out.tuple(), *a.tuple())
            return out

        return _map_fn

    @staticmethod
    def zip(
        fn: Callable[[float, float], float]
    ) -> Callable[[t.Tensor, t.Tensor], t.Tensor]:
        """
        Higher order tensor zip function.
        """
        zip_fn = tensor_zip(njit()(fn))

        def _zip_fn(a: t.Tensor, b: t.Tensor) -> t.Tensor:
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
    ) -> Callable[[t.Tensor, int], t.Tensor]:
        """
        Higher order tensor reduce function.
        """

        reduce_fn = tensor_reduce(njit()(fn))

        def _reduce(a: t.Tensor, dim: int) -> t.Tensor:
            # Set dimension that will be reduce to 1.
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = a.make(
                storage=np.ones(np.prod(out_shape)) * start,
                shape=tuple(out_shape),
                backend=a.backend,
            )
            reduce_fn(*out.tuple(), *a.tuple(), dim)
            return out

        return _reduce

    @staticmethod
    def matrix_multiply(x: t.Tensor, y: t.Tensor) -> t.Tensor:
        raise NotImplementedError
