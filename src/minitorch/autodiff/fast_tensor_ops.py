from __future__ import annotations

from typing import Callable

import numpy as np
from numba import njit, prange

from minitorch.autodiff import Tensor

from .tensor_data import (
    MAX_DIMS,
    Index,
    Shape,
    Storage,
    Strides,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

# JIT compiled functions
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)
shape_broadcast = njit(inline="always")(shape_broadcast)


def tensor_map(fn: Callable[[float], float]):
    ...


def tensor_zip(fn: Callable[[float, float], float]):
    ...


def tensor_reduce(fn: Callable[[float, float], float]):
    ...


def tensor_matrix_multiply():
    ...


class FastTensorOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        ...

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        ...

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0
    ) -> Callable[[Tensor, int], Tensor]:
        ...

    @staticmethod
    def matrix_multiply(x: Tensor, y: Tensor) -> Tensor:
        ...
