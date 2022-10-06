"""
Implementations of the autodifferentiation Functions for Tensors.
"""

from __future__ import annotations

import random
from abc import abstractmethod
from typing import Any, ForwardRef, Iterable, List, Tuple

import minitorch.operators as operators
from minitorch.autodiff import Context, History
from minitorch.autodiff.utils import wrap_tuple

from .tensor import TENSOR_COUNT, Tensor
from .tensor_data import Index, Shape
from .tensor_ops import SimpleBackend, TensorBackend


class BaseTensorFunction:

    # TODO: check that the implementation here matches ScalarBaseFunction
    #  can I actually inherit from Function and write tensor functions?

    @classmethod
    @abstractmethod
    def backward(cls, ctx: Context, grad_out: Tensor) -> Tensor:
        ...

    @classmethod
    @abstractmethod
    def forward(cls, ctx: Context, *inputs: Iterable[Tensor]):
        ...

    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))

    @classmethod
    def _forward(cls, ctx: Context, *inputs: Iterable[Tensor]) -> Tensor:
        return cls.forward(ctx, *inputs)

    @classmethod
    def apply(cls, *tensors: Iterable[Tensor]) -> Tensor:
        raw_values = []
        requires_grad = False
        for t in tensors:
            if t.history is not None:
                requires_grad = True
            raw_values.append(t.detach())

        # Create new context for Tensor
        ctx = Context(requires_grad)

        # Call forward with input values
        c = cls._forward(ctx, *raw_values)

        back = History(last_fn=cls, ctx=ctx, inputs=tensors) if requires_grad else None
        return Tensor(c.data, back, backend=c.backend)


class Neg(BaseTensorFunction):
    @classmethod
    def forward(cls, ctx: Context, a: Tensor) -> Tensor:
        return a.func.neg_map(a)

    @classmethod
    def backward(cls, ctx: Context, grad_out: Tensor) -> Tensor:
        return grad_out.func.neg_map(grad_out)


class Inv(BaseTensorFunction):
    @classmethod
    def forward(cls, ctx: Context, a: Tensor) -> Tensor:
        ctx.save_for_backward(a)
        return a.func.inv_map(a)

    @classmethod
    def backward(cls, ctx: Context, grad_out: Tensor) -> Tensor:
        a = ctx.saved_values
        return grad_out.func.inv_diff_zip(a, grad_out)


class Add(BaseTensorFunction):
    @classmethod
    def forward(cls, ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        return a.func.add_zip(a, b)

    @classmethod
    def backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, Tensor]:
        return grad_out, grad_out


class Mul(BaseTensorFunction):
    @classmethod
    def forward(cls, ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        return a.func.mul_zip(a, b)

    @classmethod
    def backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, Tensor]:
        (a, b) = ctx.saved_values
        return grad_out.func.mul_zip(b, grad_out), grad_out.func.mul_zip(a, grad_out)


class Sigmoid(BaseTensorFunction):
    @classmethod
    def forward(cls, ctx: Context, a: Tensor) -> Tensor:
        ctx.save_for_backward(a)
        return a.func.sigmoid_map(a)

    @classmethod
    def backward(cls, ctx: Context, grad_out: Tensor) -> Tensor:
        a = ctx.saved_values
        return grad_out.func.sigmoid_diff_zip(a, grad_out)


class ReLU(BaseTensorFunction):
    @classmethod
    def forward(cls, ctx: Context, a: Tensor) -> Tensor:
        ctx.save_for_backward(a)
        return a.func.relu_map(a)

    @classmethod
    def backward(cls, ctx: Context, grad_out: Tensor) -> Tensor:
        a = ctx.saved_values
        return grad_out.func.relu_diff_zip(a, grad_out)


# log
# exp
# sum
#
# all
