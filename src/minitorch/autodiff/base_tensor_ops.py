from __future__ import annotations

from abc import abstractmethod
from typing import Callable, Optional, Type

from typing_extensions import Protocol

import minitorch.autodiff.tensor as t
from minitorch import operators


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
