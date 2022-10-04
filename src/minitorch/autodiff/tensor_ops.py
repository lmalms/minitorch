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
