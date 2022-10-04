"""
Implementation of the code Tensor object for autodifferentiation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np

from minitorch import operators
from minitorch.types import TensorLike

from .tensor_data import Index, Shape, Storage, Strides, TensorData, _Shape, _Strides
from .variable import BaseFunction, Context, Variable, backpropagate

TENSOR_COUNT = 0


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


class Tensor(Variable):
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
        super().__init__(history=history, name=name)
        self.data = data
        self.backend = backend

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
    def derivative(self):
        # TODO: validate types here!
        return self._derivative

    @derivative.setter
    def derivative(self, value: TensorLike) -> None:
        """
        Validates derivative type before setting attribute.
        """
        # TODO: validate types here!
        if not isinstance(value, (int, float, Tensor)):
            raise TypeError(
                f"Derivatives have to be of type int or float - got {type(value)}."
            )
        self._derivative = value

    @property
    def grad(self):
        """
        Alias for derivative.
        """
        return self._derivative

    @grad.setter
    def grad(self, value: TensorLike) -> None:
        """
        Alias for derivative setter.
        """
        self.derivative = value

    @property
    def shape(self) -> Shape:
        return self.data.shape

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dims(self) -> int:
        return self.data.dims

    @property
    def parents(self) -> Iterable[Variable]:
        assert not self.is_constant()
        return self.history.inputs

    @staticmethod
    def _format_variable_id() -> str:
        global TENSOR_COUNT
        TENSOR_COUNT += 1
        return "Tensor" + str(TENSOR_COUNT)

    def __add__(self, other: TensorLike) -> Tensor:
        raise NotImplementedError

    def __radd__(self, other: TensorLike) -> Tensor:
        raise NotImplementedError

    def __sub__(self, other: TensorLike) -> Tensor:
        raise NotImplementedError

    def __rsub__(self, other: TensorLike) -> Tensor:
        raise NotImplementedError

    def __mul__(self, other: TensorLike) -> Tensor:
        raise NotImplementedError

    def __rmul__(self, other: TensorLike) -> Tensor:
        raise NotImplementedError

    def __truediv__(self, other: TensorLike) -> Tensor:
        raise NotImplementedError

    def __rtruediv__(self, other: TensorLike) -> Tensor:
        raise NotImplementedError

    def __matmul__(self, other: TensorLike) -> Tensor:
        raise NotImplementedError

    def __lt__(self, other: TensorLike) -> Tensor:
        raise NotImplementedError

    def __gt__(self, other: TensorLike) -> Tensor:
        raise NotImplementedError

    def __eq__(self, other: TensorLike) -> Tensor:
        raise NotImplementedError

    def __ge__(self, other: TensorLike) -> Tensor:
        raise NotImplementedError

    def __le__(self, other: TensorLike) -> Tensor:
        raise NotImplementedError

    def __neg__(self, other: TensorLike) -> Tensor:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.data.to_string()

    def __getitem__(self, key: Union[int, Index]) -> float:
        if isinstance(key, int):
            key = (key,)
        return self.data.get(key)

    def __setitem__(self, key: Union[int, Index], value: float) -> None:
        if isinstance(key, int):
            key = (key,)
        self.data.set(key, value)

    def _type_(self, backend: TensorBackend) -> None:
        self.backend = backend
        if backend.cuda:
            self.data.to_cuda_()

    def _new(self, td: TensorData) -> Tensor:
        return Tensor(data=td, backend=self.backend)

    def _ensure_tensor(self, t: TensorLike) -> Tensor:
        """
        Turns a python float into a tensor with the same backend
        """
        if isinstance(t, (int, float)):
            return Tensor.make([t], (1,), backend=self.backend)

        t._type_(self.backend)
        return t

    def all(self, dim: Optional[int] = None) -> Tensor:
        raise NotImplementedError

    def is_close(self, t: Tensor) -> Tensor:
        raise NotImplementedError

    def sigmoid(self) -> Tensor:
        raise NotImplementedError

    def relu(self) -> Tensor:
        raise NotImplementedError

    def log(self) -> Tensor:
        raise NotImplementedError

    def exp(self) -> Tensor:
        raise NotImplementedError

    def square(self) -> Tensor:
        raise NotImplementedError

    def cube(self) -> Tensor:
        raise NotImplementedError

    def sum(self, dim: Optional[int] = None) -> Tensor:
        """
        Computes the sum over dimension dim
        """
        raise NotImplementedError

    def mean(self, dim: Optional[int] = None) -> Tensor:
        """
        Computes the mean over dimension dim.
        """
        raise NotImplementedError

    def permute(self, *order: Iterable[int]) -> Tensor:
        """
        Permute tensor dimensions to *order
        """
        raise NotImplementedError

    def view(self, *shape: Iterable[int]) -> Tensor:
        """
        Changes the view of the tensor to new shape with the same size.
        """
        raise NotImplementedError

    def contiguous(self) -> Tensor:
        """
        Returns a contiguous tensor with the same data.
        """
        raise NotImplementedError

    def item(self) -> float:
        assert self.size == 1
        return self[0]

    @classmethod
    def make(
        cls,
        storage: Union[Storage, List[float]],
        shape: Shape,
        strides: Optional[Strides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """
        Creates a new tensor from data.
        """
        return Tensor(
            data=TensorData(storage=storage, shape=shape, strides=strides),
            backend=backend,
        )

    def expand(self, other: Tensor) -> Tensor:
        """
        Method used to allow from backprop over broadcasting.
        This method is called when the output of backward
        is a different size than the input to forward.

        Args:
            other: backward tensor (must broadcast with self)

        Returns:
            Expanded version of other with the right derivatives.
        """
        # If shapes are equal return shape
        if self.shape == other.shape:
            return other

        # Backward is smaller -> broadcast up
        broadcast_shape = TensorData.shape_broadcast(self.shape, other.shape)
        padding = self.zeros(broadcast_shape)
        self.backend.id_map(other, padding)
        if self.shape == broadcast_shape:
            return padding

        # Still different, reduce extra dimensions.
        out = padding
        original_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if original_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape}, {self.size}"
        return Tensor.make(out.data.storage, self.shape, backend=self.backend)

    def zeros(self, shape: Optional[Shape] = None) -> Tensor:
        def zero(shape: Shape) -> Tensor:
            return Tensor.make([0.0] * self.size, shape, backend=self.backend)

        out = zero(shape if shape is not None else self.shape)
        out._type_(self.backend)
        return out

    def to_numpy(self) -> np.ndarray:
        """
        Returns:
            Tensor data as numpy array.
        """
        return self.contiguous().data.storage.reshape(self.shape)

    def tuple(self) -> Tuple[Storage, _Shape, _Strides]:
        return self.data.tuple()

    def detach(self) -> Tensor:
        return Tensor(data=self.data, history=None, backend=self.backend)

    def chain_rule():
        raise NotImplementedError

    def backward():
        raise NotImplementedError
