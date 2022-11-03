"""
Implementations of the autodifferentiation Functions for Tensors.
"""

from __future__ import annotations

import random
from abc import abstractmethod
from typing import Any, Callable, Iterable, List, Tuple, Union

import minitorch.autodiff.tensor as t
import minitorch.functional as f
from minitorch import operators

from .tensor_data import Index, Shape, Storage, TensorData
from .tensor_ops import SimpleBackend, TensorBackend
from .utils import wrap_tuple
from .variable import BaseFunction, Context, is_constant


class TensorFunction(BaseFunction):
    @classmethod
    def to_data_type(cls, value: Any):
        """
        If value is not already of type Tensor, tries to convert.
        """
        if isinstance(value, t.Tensor):
            return value

        value = tensor(value)
        if isinstance(value, t.Tensor):
            return value

        raise ValueError("Cannot convert value to Tensor.")

    @classmethod
    def backward(cls, ctx: Context, grad_out: t.Tensor) -> Tuple[t.Tensor, ...]:
        return wrap_tuple(cls._backward(ctx, grad_out))

    @classmethod
    def forward(cls, ctx: Context, *inputs: t.Tensor) -> t.Tensor:
        return cls.to_data_type(cls._forward(ctx, *inputs))

    @classmethod
    def variable(
        cls, data: TensorData, history: t.TensorHistory, backend: TensorBackend
    ):
        return t.Tensor(data, history=history, backend=backend)

    @classmethod
    def apply(cls, *tensors: t.Tensor) -> t.Tensor:
        raw_values = []
        requires_grad = False
        for tensor in tensors:
            if tensor.history is not None:
                requires_grad = True
            raw_values.append(tensor.detach())

        # Create new context for Tensor
        ctx = Context(requires_grad)

        # Call forward with input values
        c = cls.forward(ctx, *raw_values)

        back = (
            t.TensorHistory(last_fn=cls, ctx=ctx, inputs=tensors)
            if requires_grad
            else None
        )

        return cls.variable(c.data, back, c.backend)

    @classmethod
    def chain_rule(
        cls, ctx: Context, inputs: Iterable[Union[t.Tensor, float]], grad_out: t.Tensor
    ):
        gradients = cls.backward(ctx, grad_out)
        if len(gradients) != len(inputs):
            raise IndexError("Expecting a gradient for each input.")
        tensor_grad_pairs = list(zip(inputs, gradients))
        tensor_grad_pairs = [
            (t_, t_.expand(grad))
            for (t_, grad) in tensor_grad_pairs
            if not is_constant(t_)
        ]
        return tensor_grad_pairs

    @classmethod
    @abstractmethod
    def _backward(cls, ctx: Context, grad_out: t.Tensor) -> Tuple[t.Tensor, ...]:
        ...

    @classmethod
    @abstractmethod
    def _forward(cls, ctx: Context, *inputs: t.Tensor) -> t.Tensor:
        ...


class Neg(TensorFunction):
    @classmethod
    def _forward(cls, ctx: Context, a: t.Tensor) -> t.Tensor:
        return a.func.neg_map(a)

    @classmethod
    def _backward(cls, ctx: Context, grad_out: t.Tensor) -> t.Tensor:
        return grad_out.func.neg_map(grad_out)


class Inv(TensorFunction):
    @classmethod
    def _forward(cls, ctx: Context, a: t.Tensor) -> t.Tensor:
        ctx.save_for_backward(a)
        return a.func.inv_map(a)

    @classmethod
    def _backward(cls, ctx: Context, grad_out: t.Tensor) -> t.Tensor:
        a = ctx.saved_tensors
        return grad_out.func.inv_diff_zip(a, grad_out)


class Add(TensorFunction):
    @classmethod
    def _forward(cls, ctx: Context, a: t.Tensor, b: t.Tensor) -> t.Tensor:
        return a.func.add_zip(a, b)

    @classmethod
    def _backward(cls, ctx: Context, grad_out: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        return grad_out, grad_out


class Mul(TensorFunction):
    @classmethod
    def _forward(cls, ctx: Context, a: t.Tensor, b: t.Tensor) -> t.Tensor:
        ctx.save_for_backward(a, b)
        return a.func.mul_zip(a, b)

    @classmethod
    def _backward(cls, ctx: Context, grad_out: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        (a, b) = ctx.saved_tensors
        return grad_out.func.mul_zip(b, grad_out), grad_out.func.mul_zip(a, grad_out)


class Sigmoid(TensorFunction):
    @classmethod
    def _forward(cls, ctx: Context, a: t.Tensor) -> t.Tensor:
        ctx.save_for_backward(a)
        return a.func.sigmoid_map(a)

    @classmethod
    def _backward(cls, ctx: Context, grad_out: t.Tensor) -> t.Tensor:
        a = ctx.saved_tensors
        return grad_out.func.sigmoid_diff_zip(a, grad_out)


class ReLU(TensorFunction):
    @classmethod
    def _forward(cls, ctx: Context, a: t.Tensor) -> t.Tensor:
        ctx.save_for_backward(a)
        return a.func.relu_map(a)

    @classmethod
    def _backward(cls, ctx: Context, grad_out: t.Tensor) -> t.Tensor:
        a = ctx.saved_tensors
        return grad_out.func.relu_diff_zip(a, grad_out)


class Log(TensorFunction):
    @classmethod
    def _forward(cls, ctx: Context, a: t.Tensor) -> t.Tensor:
        ctx.save_for_backward(a)
        return a.func.log_map(a)

    @classmethod
    def _backward(cls, ctx: Context, grad_out: t.Tensor) -> t.Tensor:
        a = ctx.saved_tensors
        return grad_out.func.log_diff_zip(a, grad_out)


class Exp(TensorFunction):
    @classmethod
    def _forward(cls, ctx: Context, a: t.Tensor) -> t.Tensor:
        ctx.save_for_backward(a)
        return a.func.exp_map(a)

    @classmethod
    def _backward(cls, ctx: Context, grad_out: t.Tensor) -> t.Tensor:
        a = ctx.saved_tensors
        return grad_out.func.mul_zip(a.func.exp_map(a), grad_out)


class Square(TensorFunction):
    @classmethod
    def _forward(cls, ctx: Context, a: t.Tensor) -> t.Tensor:
        ctx.save_for_backward(a)
        return a.func.mul_zip(a, a)

    @classmethod
    def _backward(cls, ctx: Context, grad_out: t.Tensor) -> t.Tensor:
        a = ctx.saved_tensors
        return grad_out.func.mul_zip(grad_out, a.func.mul_zip(a, a._ensure_tensor(2)))


class Sum(TensorFunction):
    @classmethod
    def _forward(cls, ctx: Context, a: t.Tensor, dim: t.Tensor) -> t.Tensor:
        ctx.save_for_backward(a.shape, dim)
        return a.func.add_reduce(a, int(dim.item()))

    @classmethod
    def _backward(cls, ctx: Context, grad_out: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        _, dim = ctx.saved_values
        return grad_out, zeros((1,))


class All(TensorFunction):
    @classmethod
    def _forward(cls, ctx: Context, a: t.Tensor, dim: t.Tensor) -> t.Tensor:
        return a.func.mul_reduce(a, int(dim.item()))

    @classmethod
    def _backward(cls, ctx: Context, grad_out: t.Tensor) -> t.Tensor:
        raise NotImplementedError


class LT(TensorFunction):
    @classmethod
    def _forward(cls, ctx: Context, a: t.Tensor, b: t.Tensor) -> t.Tensor:
        ctx.save_for_backward(a.shape, b.shape)
        return a.func.lt_zip(a, b)

    @classmethod
    def _backward(cls, ctx: Context, grad_out: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        a_shape, b_shape = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)


class GT(TensorFunction):
    @classmethod
    def _forward(cls, ctx: Context, a: t.Tensor, b: t.Tensor) -> t.Tensor:
        ctx.save_for_backward(a.shape, b.shape)
        return a.func.gt_zip(a, b)

    @classmethod
    def _backward(cls, ctx: Context, grad_out: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        a_shape, b_shape = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)


class LE(TensorFunction):
    @classmethod
    def _forward(cls, ctx: Context, a: t.Tensor, b: t.Tensor) -> t.Tensor:
        ctx.save_for_backward(a.shape, b.shape)
        return a.func.le_zip(a, b)

    @classmethod
    def _backward(cls, ctx: Context, grad_out: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        a_shape, b_shape = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)


class GE(TensorFunction):
    @classmethod
    def _forward(cls, ctx: Context, a: t.Tensor, b: t.Tensor) -> t.Tensor:
        ctx.save_for_backward(a.shape, b.shape)
        return a.func.ge_zip(a, b)

    @classmethod
    def _backward(cls, ctx: Context, grad_out: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        a_shape, b_shape = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)


class EQ(TensorFunction):
    @classmethod
    def _forward(cls, ctx: Context, a: t.Tensor, b: t.Tensor) -> t.Tensor:
        ctx.save_for_backward(a.shape, b.shape)
        return a.func.eq_zip(a, b)

    @classmethod
    def _backward(cls, ctx: Context, grad_out: t.Tensor) -> t.Tensor:
        a_shape, b_shape = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)


class IsClose(TensorFunction):
    @classmethod
    def _forward(cls, ctx: Context, a: t.Tensor, b: t.Tensor) -> t.Tensor:
        ctx.save_for_backward(a.shape, b.shape)
        return a.func.is_close_zip(a, b)

    @classmethod
    def _backward(
        cls, ctx: Context, a: t.Tensor, b: t.Tensor
    ) -> Tuple[t.Tensor, t.Tensor]:
        a_shape, b_shape = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)


class Permute(TensorFunction):
    @classmethod
    def _forward(cls, ctx: Context, a: t.Tensor, order: t.Tensor) -> t.Tensor:
        ctx.save_for_backward(order)
        return t.Tensor(data=a.data.permute(*[order[i] for i in range(order.size)]))

    @classmethod
    def _backward(cls, ctx: Context, grad_out: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        # Reverse of permute is argsort of permute order.
        order = ctx.saved_tensors
        order_list = [order[i] for i in range(order.size)]
        reverse_order, _ = zip(*sorted(enumerate(order_list), key=lambda pair: pair[1]))
        grad_out = t.Tensor(data=grad_out.data.permute(*reverse_order))
        return grad_out, zeros((1,))


class View(TensorFunction):
    @classmethod
    def _forward(cls, ctx: Context, a: t.Tensor, shape: t.Tensor) -> t.Tensor:
        ctx.save_for_backward(a.shape)
        assert a.data.is_contiguous(), "Tensor must be contiguous to view."

        # Make sure shapes are all ints.
        shape = [int(shape[i]) for i in range(shape.size)]
        return t.Tensor.make(a.data.storage, tuple(shape), backend=a.backend)

    @classmethod
    def _backward(cls, ctx: Context, grad_out: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        original_shape = ctx.saved_values
        return t.Tensor.make(
            grad_out.data.storage, original_shape, backend=grad_out.backend
        ), zeros((1,))


class Copy(TensorFunction):
    @classmethod
    def _forward(cls, ctx: Context, a: t.Tensor) -> t.Tensor:
        return a.func.id_map(a)

    @classmethod
    def _backward(cls, ctx: Context, grad_out: t.Tensor) -> t.Tensor:
        return grad_out


class MatMul(TensorFunction):
    @classmethod
    def _forward(cls, ctx: Context, a: t.Tensor, b: t.Tensor) -> t.Tensor:
        ctx.save_for_backward(a, b)
        return a.func.matrix_multiply(a, b)

    @classmethod
    def _backward(cls, ctx: Context, grad_out: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        a, b = ctx.saved_tensors

        def transpose(a: t.Tensor) -> t.Tensor:
            order = reversed(list(range(a.dims)))
            return a._new(data=a.data.permute(*order))

        return (
            grad_out.func.matrix_multiply(grad_out, transpose(b)),
            grad_out.func.matrix_multiply(transpose(a), grad_out),
        )


### Tensor utils functions ###


def zeros(shape: Shape, backend: TensorBackend = SimpleBackend) -> t.Tensor:
    return t.Tensor.make([0.0] * int(f.product(list(shape))), shape, backend=backend)


def ones(shape: Shape, backend: TensorBackend = SimpleBackend) -> t.Tensor:
    return t.Tensor.make([1.0] * int(f.product(list(shape))), shape, backend=backend)


def rand(
    shape: Shape, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
):
    vals = [random.random() for _ in range(int(f.product(list(shape))))]
    tensor = t.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad = requires_grad
    return tensor


def _tensor(
    data: Storage,
    shape: Shape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
):
    tensor = t.Tensor.make(data, shape, backend=backend)
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


def grad_central_difference(
    f: Callable[..., t.Tensor],
    *inputs: t.Tensor,
    idx: Index,
    arg: int,
    epsilon: float = 1e-06,
):
    # Get the value to compute the derivative wrt to
    x = inputs[arg]
    delta = zeros(x.shape)
    delta[idx] = epsilon
    upper = [x if (i != arg) else (x + delta) for i, x in enumerate(inputs)]
    lower = [x if (i != arg) else (x - delta) for i, x in enumerate(inputs)]
    delta = f(*upper).sum() - f(*lower).sum()
    return delta.item() / (2 * epsilon)


def grad_check(f: Callable[..., t.Tensor], *tensors: t.Tensor) -> None:

    # Compute derivatives with respect to each of the inputs.
    for tensor in tensors:
        tensor.requires_grad = True
        tensor.zero_grad_()

    out_ = f(*tensors)
    out_.sum().backward()

    for i, tensor in enumerate(tensors):
        for _ in range(10):
            # Grad a random index within that tensor imput
            idx = tensor.data.sample()
            check = grad_central_difference(f, *tensors, arg=i, idx=idx)
            assert tensor.grad is not None

            if not operators.is_close(tensor.grad[idx], check):
                raise ValueError(
                    f"Derivative check failed for function {f.__name__} with arguments {tensors}. "
                    f"Derivative failed at input position {i}, index {idx}. "
                    f"Calculated derivative is {tensor.grad[idx]}, should be {check}."
                )
