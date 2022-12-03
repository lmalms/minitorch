from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, Optional, Tuple, Union

from minitorch import operators
from .variable import BaseFunction, Context, History, Variable


class Scalar(Variable):
    """
    A reimplementation of scalar values for auto-differentiation. Scalar variables
    behave as close as possible to Python built in floats while also tracking operations that lead
    to the scalar's creation. They can only be manipulated using ScalarFunction class.

    Attributes:
        data - float
            The scalar value.
    """

    def __init__(
        self, value: float, history: History = History(), name: Optional[str] = None
    ):
        super().__init__(history=history, name=name)
        self.data = value

    def __hash__(self) -> int:
        # Hash method is not inherited if overwriting __eq__
        return hash(self.id_)

    @property
    def data(self) -> float:
        return self._data

    @data.setter
    def data(self, value: Union[int, float]) -> None:
        """
        Validates data type before setting data attribute.
        """
        if not isinstance(value, (float, int)):
            raise TypeError(
                f"Scalar values have to be of type int or float - got {type(value)}."
            )
        self._data = float(value)

    def __repr__(self) -> str:
        return f"Scalar({self.data:.3f}, name={self.name})"

    def __bool__(self):
        return bool(self._data)

    def __add__(self, other: Union[int, float, Scalar]) -> Scalar:
        return Add.apply(self, other)

    def __radd__(self, other: Union[int, float, Scalar]) -> Scalar:
        return Add.apply(self, other)

    def __sub__(self, other: Union[int, float, Scalar]) -> Scalar:
        return Add.apply(self, Neg.apply(other))

    def __rsub__(self, other: Union[int, float, Scalar]) -> Scalar:
        return Add.apply(Neg.apply(other), self)

    def __mul__(self, other: Union[int, float, Scalar]) -> Scalar:
        return Mul.apply(self, other)

    def __rmul__(self, other: Union[int, float, Scalar]) -> Scalar:
        return Mul.apply(self, other)

    def __truediv__(self, other: Union[int, float, Scalar]) -> Scalar:
        return Mul.apply(self, Inv.apply(other))

    def __rtruediv__(self, other: Union[int, float, Scalar]) -> Scalar:
        return Mul.apply(other, Inv.apply(self))

    def __lt__(self, other: Union[int, float, Scalar]) -> Scalar:
        return LT.apply(self, other)

    def __gt__(self, other: Union[int, float, Scalar]) -> Scalar:
        return GT.apply(self, other)

    def __eq__(self, other: Union[int, float, Scalar]) -> Scalar:
        return EQ.apply(self, other)

    def __ge__(self, other: Union[int, float, Scalar]) -> Scalar:
        return GE.apply(self, other)

    def __le__(self, other: Union[int, float, Scalar]) -> Scalar:
        return LE.apply(self, other)

    def __neg__(self) -> Scalar:
        return Neg.apply(self)

    def square(self) -> Scalar:
        return Square.apply(self)

    def cube(self) -> Scalar:
        return Cube.apply(self)

    def log(self) -> Scalar:
        return Log.apply(self)

    def exp(self) -> Scalar:
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        return ReLU.apply(self)


class ScalarFunction(BaseFunction):
    """
    Wrapper for a mathematical function that processes and produces Scalar variables.
    """

    @classmethod
    def to_data_type(cls, value: Any) -> float:
        return float(value)

    @classmethod
    def variable(cls, value: Union[int, float], history: History = History()) -> Scalar:
        return Scalar(value, history)

    @classmethod
    def forward(cls, ctx: Context, *values) -> float:
        """
        Forward call.
        Args:
            ctx - Context
                A context container to save any information to that may be needed for
                backward call.
            *values - List[float]
                n floats to run forward call over.
        """
        return cls.to_data_type(cls._forward(ctx, *values))

    @classmethod
    @abstractmethod
    def _forward(cls, ctx: Context, *values) -> float:
        ...

    @classmethod
    def backward(cls, ctx: Context, d_out: float) -> Union[Tuple[float, ...], float]:
        """
        Backward call.

        Args:
            ctx - Container
                A container object that holds any information recorded during the forward call.
            d_out - float
                Derivative is multiplied by this value.
        """
        ...


class Add(ScalarFunction):
    """Addition function f(x, y) = x + y"""

    @classmethod
    def _forward(cls, ctx: Context, a: float, b: float) -> float:
        return operators.add(a, b)

    @classmethod
    def backward(cls, ctx: Context, d_out: float) -> Tuple[float, float]:
        return d_out, d_out


class Log(ScalarFunction):
    """Log function f(x) = log(x)"""

    @classmethod
    def _forward(cls, ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.log(a)

    @classmethod
    def backward(cls, ctx: Context, d_out: float) -> float:
        a = ctx.saved_values
        return operators.log_diff(a, d_out)


class Mul(ScalarFunction):
    """Multiplication for Scalars: f(x, y) = x * y"""

    @classmethod
    def _forward(cls, ctx: Context, a: float, b: float) -> float:
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @classmethod
    def backward(cls, ctx: Context, d_out: float) -> Tuple[float, float]:
        a, b = ctx.saved_values
        return operators.mul(b, d_out), operators.mul(a, d_out)


class Inv(ScalarFunction):
    """Inverse function for Scalars: f(x) = 1(x)"""

    @classmethod
    def _forward(cls, ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.inv(a)

    @classmethod
    def backward(cls, ctx: Context, d_out: float) -> float:
        a = ctx.saved_values
        return operators.inv_diff(a, d_out)


class Neg(ScalarFunction):
    """Negation function for Scalars: f(x) = -x"""

    @classmethod
    def _forward(cls, ctx: Context, a: float) -> float:
        return operators.neg(a)

    @classmethod
    def backward(cls, ctx: Context, d_out: float) -> float:
        return operators.mul(-1, d_out)


class Square(ScalarFunction):
    """Squaring function to sclars f(x) = x ** 2"""

    @classmethod
    def _forward(cls, ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.mul(a, a)

    @classmethod
    def backward(cls, ctx: Context, d_out: float) -> float:
        a = ctx.saved_values
        return operators.mul(d_out, operators.mul(2, a))


class Cube(ScalarFunction):
    """Cube function to scalars f(x) = x ** 3"""

    @classmethod
    def _forward(cls, ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.mul(a, operators.mul(a, a))

    @classmethod
    def backward(cls, ctx: Context, d_out: float) -> float:
        a = ctx.saved_values
        return operators.mul(d_out, operators.mul(3, a))


class Sigmoid(ScalarFunction):
    """Sigmoid function applied to Scalars: f(x) = 1. / (1. + e^-x)"""

    @classmethod
    def _forward(cls, ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @classmethod
    def backward(cls, ctx: Context, d_out: float) -> float:
        a = ctx.saved_values
        return operators.sigmoid_diff(a, d_out)


class ReLU(ScalarFunction):
    """ReLU function applied to Scalars: f(x) = relu(x)"""

    @classmethod
    def _forward(cls, ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.relu(a)

    @classmethod
    def backward(cls, ctx: Context, d_out: float) -> float:
        a = ctx.saved_values
        return operators.relu_diff(a, d_out)


class Exp(ScalarFunction):
    """exp function applied to Scalars: f(x) = exp(x)"""

    @classmethod
    def _forward(cls, ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.exp(a)

    @classmethod
    def backward(cls, ctx: Context, d_out: float) -> float:
        a = ctx.saved_values
        return d_out * operators.exp(a)


class LT(ScalarFunction):
    """Less than function on scalars: f(x, y) = 1.0 if x < y else 0."""

    @classmethod
    def _forward(cls, ctx: Context, a: float, b: float) -> float:
        return operators.lt(a, b)

    @classmethod
    def backward(cls, ctx: Context, d_out: float) -> Tuple[float, float]:
        return 0.0, 0.0


class GT(ScalarFunction):
    """Greater than function for scalars: f(x, y) = 1. if x > y else 0."""

    @classmethod
    def _forward(cls, ctx: Context, a: float, b: float) -> float:
        return operators.gt(a, b)

    @classmethod
    def backward(cls, ctx: Context, d_out: float) -> Tuple[float, float]:
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equality function on scalars: f(x, y) = 1. if x == y else 0."""

    @classmethod
    def _forward(cls, ctx: Context, a: float, b: float) -> float:
        return operators.eq(a, b)

    @classmethod
    def backward(cls, ctx: Context, d_out: float) -> Tuple[float, float]:
        return 0.0, 0.0


class GE(ScalarFunction):
    """Greater than or equal function for scalars: f(x, y) = 1. if x >= y else 0."""

    @classmethod
    def _forward(cls, ctx: Context, a: float, b: float) -> float:
        return operators.ge(a, b)

    @classmethod
    def backward(cls, ctx: Context, d_out: float) -> Tuple[float, float]:
        return 0.0, 0.0


class LE(ScalarFunction):
    """Less than or equal function for scalars: f(x, y) = 1. if x <= y else 0."""

    @classmethod
    def _forward(cls, ctx: Context, a: float, b: float) -> float:
        return operators.le(a, b)

    @classmethod
    def backward(cls, ctx: Context, d_out: float) -> Tuple[float, float]:
        return 0.0, 0.0


def central_difference(
    func: Callable[..., Scalar], *values, arg_idx: int = 0, epsilon=1e-02
) -> Scalar:
    """
    Computes a numerical approximation of the derivative of f with respect to one arg.

    Args:
        func - Callable[..., Any]
            The function to differentiate.
        *values - List[...]
            The parameters to pass to func.
        arg_idx - int, default = 0
            The index of the variable in *values to compute the derivative with respect to.
        epsilon - float, default = 1e-06
            A small constant.
    """
    upper_values = [
        (val + epsilon) if i == arg_idx else val for (i, val) in enumerate(values)
    ]
    lower_values = [
        (val - epsilon) if i == arg_idx else val for (i, val) in enumerate(values)
    ]

    return (func(*upper_values) - func(*lower_values)) / (2 * epsilon)


def derivative_check(func: Callable[..., Scalar], *scalars):
    """
    Checks that autodiff works on an arbitrary python function.
    Asserts False if derivative is incorrect.
    """
    for scalar in scalars:
        scalar.requires_grad = True
    out_ = func(*scalars)
    out_.backward()

    # Run derivative check using central_difference
    for (i, scalar) in enumerate(scalars):
        check = central_difference(func, *scalars, arg_idx=i)

        if not operators.is_close(scalar.derivative, check.data):
            raise ValueError(
                f"Derivative check failed for function {func.__name__} with arguments {scalars}. "
                f"Derivative failed at position {i}. Calculated derivative is {scalar.derivative},"
                f" should be {check.data}."
            )
