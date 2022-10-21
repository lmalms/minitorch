from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy
from typing import Any, Callable, Iterable, Optional, Tuple, Type

import numpy as np
from typing_extensions import Protocol

import minitorch.autodiff.tensor as t
from minitorch import operators
from minitorch.functional import product, reduce

from .tensor_data import (
    Shape,
    Storage,
    Strides,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)


def to_numpy(*inputs: Tuple[float]) -> Iterable[np.ndarray]:
    return [np.array(in_) for in_ in inputs]


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
        out_shape, out_strides = to_numpy(out_shape, out_strides)
        in_shape, in_strides = to_numpy(in_shape, in_strides)

        # Placeholders to use during map
        out_size = int(product(in_shape.tolist()))
        in_index = np.zeros_like(in_shape)
        out_index = np.zeros_like(out_shape)

        for out_position in range(out_size):
            # Get index corresponding to ordinal in out_tensor
            to_index(out_position, out_shape, out_index)

            # Get corresponding index in possibly smaller in_tensor
            broadcast_index(out_index, out_shape, in_shape, in_index)

            # Get corrsponding ordinal into smaller in_tensor
            in_position = index_to_position(in_index, in_strides)

            # Apply func at positions
            print(in_storage[in_position])
            print(out_position)
            out_storage[out_position] = fn(in_storage[in_position])

    return _map


def tensor_zip(fn: Callable[[float, float], float]):
    """
    Low level implementation of tensor zip between
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
        # Cast to numpy arrays
        out_shape, out_strides = to_numpy(out_shape, out_strides)
        a_shape, a_strides = to_numpy(a_shape, a_strides)
        b_shape, b_strides = to_numpy(b_shape, b_strides)

        # Placeholders to fill during zip
        out_size = int(product(out_shape.tolist()))
        out_index = np.zeros_like(out_shape)
        a_index, b_index = np.zeros_like(a_shape), np.zeros_like(b_shape)

        for out_position in range(out_size):

            # Grab the index in out from position
            to_index(out_position, out_shape, out_index)

            # Get the corresponding positions in possibly smaller in_tensors
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            # From these indices get positions
            a_position = index_to_position(a_index, a_strides)
            b_position = index_to_position(b_index, b_strides)

            out_storage[out_position] = fn(a_storage[a_position], b_storage[b_position])

    return _zip


def tensor_reduce(fn: Callable[[float, float], float]) -> Any:
    """
    Low-level implementation of tensor reduce.
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
        out_shape, out_strides = to_numpy(out_shape, out_strides)
        in_shape, in_strides = to_numpy(in_shape, in_strides)

        # Placeholders to fill during reduce.
        out_size = int(product(out_shape.tolist()))
        out_index = np.zeros_like(out_shape)

        for out_position in range(out_size):
            # Grab the corresponding out_index
            to_index(out_position, out_shape, out_index)

            # Get all positions that will be reduced to that out_index
            in_positions = []
            in_index = deepcopy(out_index)
            for j in range(in_shape[reduce_dim]):
                in_index[reduce_dim] = j
                in_positions.append(index_to_position(in_index, in_strides))

            # Get all of the corresponding values
            in_values = [in_storage[j] for j in in_positions]
            out_storage[out_position] = reduce(fn, out_storage[out_position])(in_values)

    return _reduce


class MapProto(Protocol):
    def __call__(self, a: t.Tensor, out: Optional[t.Tensor] = None) -> t.Tensor:
        ...


class TensorOps:

    cuda = False

    @staticmethod
    @abstractmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        ...

    @staticmethod
    @abstractmethod
    def cmap(fn: Callable[[float], float]) -> Callable[[t.Tensor, t.Tensor], t.Tensor]:
        ...

    @staticmethod
    @abstractmethod
    def zip(
        fn: Callable[[float, float], float]
    ) -> Callable[[t.Tensor, t.Tensor], t.Tensor]:
        ...

    @staticmethod
    @abstractmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[t.Tensor, int], t.Tensor]:
        ...

    @staticmethod
    @abstractmethod
    def matrix_multiply(x: t.Tensor, y: t.Tensor) -> t.Tensor:
        ...


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """
        Dynamically constructs a tensor backend based on a TensorOps object
        that implements map, zip and reduce higher-order functions.

        Args:
            ops: tensor opertions object

        Returns:
            a collection of tensor functions
        """

        # Maps
        self.id_map = ops.map(operators.identity)
        self.id_cmap = ops.cmap(operators.identity)
        self.neg_map = ops.map(operators.neg)
        self.inv_map = ops.map(operators.inv)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.le_zip = ops.zip(operators.le)
        self.gt_zip = ops.zip(operators.gt)
        self.ge_zip = ops.zip(operators.ge)
        self.eq_zip = ops.zip(operators.eq)
        self.max_map = ops.zip(operators.maximum)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_diff_zip = ops.zip(operators.relu_diff)
        self.sigmoid_diff_zip = ops.zip(operators.sigmoid_diff)
        self.log_diff_zip = ops.zip(operators.log_diff)
        self.inv_diff_zip = ops.zip(operators.inv_diff)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """
        Higher order tensor map function.
        """
        map_fn = tensor_map(fn)

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
        zip_fn = tensor_zip(fn)

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
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[t.Tensor, int], t.Tensor]:
        """
        Higher order tensor reduce function.
        """

        reduce_fn = tensor_reduce(fn)

        def _reduce(a: t.Tensor, dim: int) -> t.Tensor:
            # Set dimension that is reduced to 1.
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = a.make(
                storage=np.ones((int(product(out_shape)))) * start,
                shape=tuple(out_shape),
                backend=a.backend,
            )
            reduce_fn(*out.tuple(), *a.tuple(), dim)
            return out

        return _reduce

    @staticmethod
    def matrix_multiply(x: t.Tensor, y: t.Tensor) -> t.Tensor:
        raise NotImplementedError


SimpleBackend = TensorBackend(SimpleOps)
