"""
Implementation of the code Tensor object for autodifferentiation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple, Type, Union

import numpy as np

import minitorch.autodiff.tensor_functions as tf
import minitorch.functional as f
from minitorch.types import TensorLike

from .tensor_data import Index, Shape, Storage, Strides, TensorData
from .tensor_ops import SimpleBackend, TensorBackend
from .variable import BaseFunction, Context, History, Variable, backpropagate

TENSOR_COUNT = 0


class TensorHistory(History):
    """
    History stores the history of a Function operations that was used
    to create the new variable
    """

    def __init__(
        self,
        last_fn: Optional[Type[BaseFunction]] = None,
        ctx: Optional[Context] = None,
        inputs: Optional[Iterable[Tensor]] = None,
    ):
        super().__init__(last_fn, ctx, inputs)

    def backprop_step(self, grad_out: Tensor) -> List[Tuple[Variable, float]]:
        """
        Runs one step of back-propagation by calling the chain rule.
        """
        return self.last_fn.chain_rule(self.ctx, self.inputs, d_out=grad_out)


class Tensor(Variable):
    """
    A generalisation of Scalar in that it is a Variable that handles multidimensional arrays.
    """

    def __init__(
        self,
        data: TensorData,
        history: Optional[TensorHistory] = None,
        backend: TensorBackend = SimpleBackend,
        name: Optional[str] = None,
    ):
        super().__init__(history=history, name=name)
        self.data = data
        self.backend = backend
        self.func = backend

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
    def requires_grad(self) -> bool:
        return self.history is not None

    @requires_grad.setter
    def requires_grad(self, requires_grad: bool) -> None:
        """
        Sets the requires_grad_ flag to 'val' on variable.
        This ensures that operations on this variable will trigger
        backpropagation.
        """
        self.history = TensorHistory() if requires_grad else None

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

    def __hash__(self):
        return hash(self.name)

    def __add__(self, other: TensorLike) -> Tensor:
        return tf.Add.apply(self, self._ensure_tensor(other))

    def __radd__(self, other: TensorLike) -> Tensor:
        return tf.Add.apply(self, self._ensure_tensor(other))

    def __sub__(self, other: TensorLike) -> Tensor:
        return tf.Add.apply(self, tf.Neg.apply(self._ensure_tensor(other)))

    def __rsub__(self, other: TensorLike) -> Tensor:
        return tf.Add.apply(self._ensure_tensor(other), tf.Neg.apply(self))

    def __mul__(self, other: TensorLike) -> Tensor:
        return tf.Mul.apply(self, self._ensure_tensor(other))

    def __rmul__(self, other: TensorLike) -> Tensor:
        return tf.Mul.apply(self, self._ensure_tensor(other))

    def __truediv__(self, other: TensorLike) -> Tensor:
        return tf.Mul.apply(self, tf.Inv.apply(self._ensure_tensor(other)))

    def __rtruediv__(self, other: TensorLike) -> Tensor:
        return tf.Mul.apply(self._ensure_tensor(other), tf.Inv.apply(self))

    def __matmul__(self, other: TensorLike) -> Tensor:
        return tf.MatMul.apply(self, other)

    def __lt__(self, other: TensorLike) -> Tensor:
        return tf.LT.apply(self, self._ensure_tensor(other))

    def __gt__(self, other: TensorLike) -> Tensor:
        return tf.GT.apply(self, self._ensure_tensor(other))

    def __eq__(self, other: TensorLike) -> Tensor:
        return tf.EQ.apply(self, self._ensure_tensor(other))

    def __ge__(self, other: TensorLike) -> Tensor:
        return tf.GE.apply(self, self._ensure_tensor(other))

    def __le__(self, other: TensorLike) -> Tensor:
        return tf.LE.apply(self, self._ensure_tensor(other))

    def __neg__(self) -> Tensor:
        return tf.Neg.apply(self)

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
        if dim is None:
            return tf.All.apply(self.view(self.size), self._ensure_tensor(0))
        return tf.All.apply(self, self._ensure_tensor(dim))

    def is_close(self, t: Tensor) -> Tensor:
        return tf.IsClose.apply(self, self._ensure_tensor(t))

    def sigmoid(self) -> Tensor:
        return tf.Sigmoid.apply(self)

    def relu(self) -> Tensor:
        return tf.ReLU.apply(self)

    def log(self) -> Tensor:
        return tf.Log.apply(self)

    def exp(self) -> Tensor:
        return tf.Exp.apply(self)

    def square(self) -> Tensor:
        raise NotImplementedError

    def cube(self) -> Tensor:
        raise NotImplementedError

    def sum(self, dim: Optional[int] = None) -> Tensor:
        """
        Computes the sum over dimension dim
        """
        if dim is None:
            return tf.Sum.apply(
                self.contiguous().view(self.size), self._ensure_tensor(0)
            )
        return tf.Sum.apply(self, self._ensure_tensor(dim))

    def mean(self, dim: Optional[int] = None) -> Tensor:
        """
        Computes the mean over dimension dim.
        """
        if dim is None:
            return self.sum() / self.size
        return self.sum(dim) / self.shape[dim]

    def permute(self, *order: int) -> Tensor:
        """
        Permute tensor dimensions to *order
        """
        return tf.Permute.apply(self, tf.tensor(list(order)))

    def view(self, *shape: int) -> Tensor:
        """
        Changes the view of the tensor to new shape with the same size.
        """
        return tf.View.apply(self, tf.tensor(list(shape)))

    def contiguous(self) -> Tensor:
        """
        Returns a contiguous tensor with the same data.
        """
        return tf.Copy.apply(self)

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
            data=TensorData(storage=storage, shape=tuple(shape), strides=strides),
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
            return Tensor.make(
                [0.0] * int(f.product(list(shape))), shape, backend=self.backend
            )

        out = zero(shape if shape is not None else self.shape)
        out._type_(self.backend)
        return out

    def to_numpy(self) -> np.ndarray:
        """
        Returns:
            Tensor data as numpy array.
        """
        return self.contiguous().data.storage.reshape(self.shape)

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        return self.data.tuple()

    def detach(self) -> Tensor:
        return Tensor(data=self.data, history=None, backend=self.backend)

    def chain_rule(self, d_out: Any) -> Iterable[Tuple[Variable, Any]]:
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn.backward(h.ctx, d_out)
        assert len(x) == len(
            h.inputs
        ), f"Bug in TensorFunction {h.last_fn}. Need a derivative for each input."
        return [
            (in_, in_.expand(self._ensure_tensor(d_in)))
            for in_, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_out: Optional[Tensor] = None) -> None:
        if grad_out is None:
            assert self.shape == (
                1,
            ), "Must provide a grad_out if non-scalar like tensor"
            grad_out = Tensor.make([1.0], shape=(1,), backend=self.backend)
        backpropagate(self, grad_out)

    def accumulate_derivative(self, value: TensorLike) -> None:
        """
        Add value to the accumulated derivative for this variable
        Should only be called on leaf variables during autodifferentiation.
        """
        return super().accumulate_derivative(value=value)
