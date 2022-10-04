from __future__ import annotations

from typing import Any, Callable, Optional, Type, _ProtocolMeta

import numpy as np
from typing_extensions import Protocol

from minitorch import operators

from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Storage,
    _Index,
    _Shape,
    _Strides,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        ...


class TensorOps:

    cuda = False

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        pass

    @staticmethod
    def cmap(fn: Callable[[float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        pass

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        pass

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        pass

    @staticmethod
    def matrix_multiply(x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError


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
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_diff_zip = ops.zip(operators.relu_diff)
        self.sigmoid_diff_zip = ops.zip(operators.sigmoid_diff)
        self.log_diff_zip = ops.zip(operators.log_diff)
        self.inv_diff_zip = ops.zip(operators.inv_diff)
