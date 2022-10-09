"""
Implementations of the autodifferentiation Functions for Tensors.
"""

from __future__ import annotations

import random
from abc import abstractmethod
from typing import Any, Callable, ForwardRef, List, Tuple, Union

import minitorch.functional as f
import minitorch.operators as operators
from minitorch.autodiff import Context, History
from minitorch.autodiff.utils import wrap_tuple
from minitorch.autodiff.variable import BaseFunction

from .tensor import TENSOR_COUNT, Tensor
from .tensor_data import Index, Shape, Storage
from .tensor_ops import SimpleBackend, TensorBackend


class BaseTensorFunction:

    # TODO: check that the implementation here matches ScalarBaseFunction
    #  can I actually inherit from Function and write tensor functions?

    @classmethod
    @abstractmethod
    def backward(
        cls, ctx: Context, grad_out: Tensor
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        ...

    @classmethod
    @abstractmethod
    def forward(cls, ctx: Context, *inputs: Tensor) -> Tensor:
        ...

    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))

    @classmethod
    def _forward(cls, ctx: Context, *inputs: Tensor) -> Tensor:
        return cls.forward(ctx, *inputs)

    @classmethod
    def apply(cls, *tensors: Tensor) -> Tensor:
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


class Log(BaseTensorFunction):
    @classmethod
    def forward(cls, ctx: Context, a: Tensor) -> Tensor:
        ctx.save_for_backward(a)
        return a.func.log_map(a)

    @classmethod
    def backward(cls, ctx: Context, grad_out: Tensor) -> Tensor:
        return super().backward(ctx, grad_out)


class Exp(BaseTensorFunction):
    @classmethod
    def forward(cls, ctx: Context, a: Tensor) -> Tensor:
        ctx.save_for_backward(a)
        return a.func.exp_map(a)

    @classmethod
    def backward(cls, ctx: Context, grad_out: Tensor) -> Tensor:
        a = ctx.save_for_backward
        return grad_out.func.mul_zip(a.func.exp_map(a), grad_out)


class Sum(BaseTensorFunction):
    @classmethod
    def forward(cls, ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape, dim)
        return a.func.add_reduce(a, int(dim.item()))

    @classmethod
    def backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, Tensor]:
        a_shape, dim = ctx.saved_values
        return (grad_out, zeros((dim,)))


class All(BaseTensorFunction):
    @classmethod
    def forward(cls, ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        return a.func.mul_reduce(a, int(dim.item()))

    @classmethod
    def backward(cls, ctx: Context, grad_out: Tensor) -> Tensor:
        raise NotImplementedError


class LT(BaseTensorFunction):
    @classmethod
    def forward(cls, ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape, b.shape)
        return a.func.lt_zip(a, b)

    @classmethod
    def backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, Tensor]:
        (
            a_shape,
            b_shape,
        ) = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)


class GT(BaseTensorFunction):
    @classmethod
    def forward(cls, ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape, b.shape)
        return a.func.gt_zip(a, b)

    @classmethod
    def backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, Tensor]:
        a_shape, b_shape = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)


class LE(BaseTensorFunction):
    @classmethod
    def forward(cls, ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape, b.shape)
        return a.func.le_zip(a, b)

    @classmethod
    def backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, Tensor]:
        a_shape, b_shape = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)


class GE(BaseTensorFunction):
    @classmethod
    def forward(cls, ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape, b.shape)
        return a.func.ge_zip(a, b)

    @classmethod
    def backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, Tensor]:
        a_shape, b_shape = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)


class EQ(BaseTensorFunction):
    @classmethod
    def forward(cls, ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape, b.shape)
        return a.func.eq_zip(a, b)

    @classmethod
    def backward(cls, ctx: Context, grad_out: Tensor) -> Tensor:
        a_shape, b_shape = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)


class IsClose(BaseTensorFunction):
    @classmethod
    def forward(cls, ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        return a.func.is_close_zip(a, b)

    @classmethod
    def backward(cls, ctx: Context, a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
        a, b = ctx.saved_values
        return zeros(a.shape), zeros(b.shape)


class Permute(BaseFunction):
    @classmethod
    def forward(cls, ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        ctx.save_for_backward(order)
        return Tensor(data=a.data.permute(*[order[i] for i in range(order.dims)]))

    @classmethod
    def backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, Tensor]:
        order = ctx.saved_values
        return Tensor(
            data=grad_out.data.permute(*[order[i] for i in range(order.dims)])
        ), zeros((1,))


class View(BaseTensorFunction):
    @classmethod
    def forward(cls, ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape)
        assert a.data.is_contiguous(), "Tensor must be contiguous to view."
        new_shape = [int(shape[i]) for i in range(shape.dims)]
        return Tensor.make(a.data.storage, tuple(new_shape), backend=a.backend)

    @classmethod
    def backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, Tensor]:
        (original_shape,) = ctx.saved_values
        return Tensor.make(
            grad_out.data.storage, original_shape, backend=grad_out.backend
        ), zeros((1,))


class Copy(BaseTensorFunction):
    @classmethod
    def forward(cls, ctx: Context, a: Tensor) -> Tensor:
        return a.func.id_map(a)

    @classmethod
    def backward(cls, ctx: Context, grad_out: Tensor) -> Tensor:
        return grad_out


class MatMul(BaseTensorFunction):
    @classmethod
    def forward(cls, ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        return a.func.matrix_multiply(a, b)

    @classmethod
    def backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, Tensor]:
        a, b = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = reversed(list(range(a.dims)))
            return a._new(data=a.data.permute(*order))

        return (
            grad_out.func.matrix_multiply(grad_out, transpose(b)),
            grad_out.func.matrix_multiply(transpose(a), grad_out),
        )


### Tensor utils functions ###


def zeros(shape: Shape, backend: TensorBackend = SimpleBackend) -> Tensor:
    return Tensor.make([0.0] * int(f.product(shape)), shape, backend=backend)


def rand(
    shape: Shape, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
):
    vals = [random.random() for _ in range((f.product(shape)))]
    tensor = Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad = requires_grad
    return tensor


def _tensor(
    data: Storage,
    shape: Shape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
):
    tensor = Tensor.make(data, shape, backend)
    tensor.requires_grad = requires_grad
    return tensor


def tensor(
    data: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
):
    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        return [ls]

    data_, shape_ = flatten(data), shape(data)
    return _tensor(data_, tuple(shape_), backend, requires_grad)


### Gradient checks for tensors ###

# TODO: check that implementation matches Scalar implementation


def grad_central_difference(
    f: Callable[..., Tensor],
    *inputs: Tensor,
    arg: int = 0,
    epsilon: float = 1e-06,
    idx: Index,
):
    # Get the value to compute the derivative wrt to
    x = inputs[arg]
    delta = zeros(x.shape)
    delta[idx] = epsilon
    upper = [x if (i != arg) else (x + delta) for i, x in enumerate(inputs)]
    lower = [x if (i != arg) else (x - delta) for i, x in enumerate(inputs)]
    delta = f(*upper).sum() - f(*lower).sum()
    return delta.item() / (2 * epsilon)


def grad_check(f: Callable[..., Tensor], *inputs: Tensor) -> None:

    # Compute derivatives with respect to each of the inputs.
    for x in inputs:
        x.requires_grad = True
        x.zero_grad_()

    random.seed(10)
    out_ = f(*inputs)
    out_.sum().backward()

    for i, in_ in enumerate(inputs):
        # Grad a random index within that tensor imput
        idx = x.data.sample()
        check = grad_central_difference(f, *inputs, arg=i, idx=idx)

        assert x.grad is not None

        # TODO: Check that grads are close.
        raise NotImplementedError
