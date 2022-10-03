"""
Implementation of the code Tensor object for autodifferentiation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np

from minitorch import operators
from minitorch.types import TensorLike

from .tensor_data import Index, Shape, Storage, Strides, TensorData, _Shape, _Strides
from .variable import BaseFunction, Context, Variable, backpropagate

TENSOR_COUNT = 0


def format_variable_id() -> str:
    global TENSOR_COUNT
    TENSOR_COUNT += 1
    return "Tensor" + str(TENSOR_COUNT)


@dataclass
class History:
    """
    History stores the history of a Function operations that was used
    to create the new variable
    """

    # TODO: Can I not just inherit / use Variable -> History here?
    last_fn: Optional[Type[BaseFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


class Tensor:
    """
    A generalisation of Scalar in that it is a Variable that handles multidimensional arrays.
    """

    def __init__(
        self,
        data: TensorData,
        history: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        self.data = data
        self.history = history
        self.id = format_variable_id()
        self.name = name if name is not None else self.id
        self.backend = backend
        self._grad = None

    @property
    def data(self) -> TensorData:
        return self._data

    @data.setter
    def data(self, data: TensorData) -> None:
        """
        Type validation before setting attribute.
        """
        if not isinstance(data, TensorData):
            raise TypeError(
                f"Data has to be of type TensorData - got type {type(data)}"
            )
        self._data = data

    @property
    def history(self):
        return self._history

    @history.setter
    def history(self, history: Optional[History] = None) -> None:
        """
        Validates history type before setting history attribute.
        """
        if not ((history is None) or isinstance(history, History)):
            raise TypeError(
                f"History has to be None or of type history - got {type(history)}"
            )
        self._history = history

    @property
    def grad(self):
        return self._grad

    # TODO: a grad setter?

    @property
    def requires_grad(self) -> bool:
        self.history is not None

    @requires_grad.setter
    def requires_grad(self, requires_grad: bool) -> None:
        self.history = History() if requires_grad else None

    @property
    def shape(self) -> Shape:
        return self.data.shape

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dims(self) -> int:
        return self.data.dims

    def to_numpy(self) -> np.ndarray:
        """
        Returns:
            Tensor data as numpy array.
        """
        pass
