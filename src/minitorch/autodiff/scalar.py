from __future__ import annotations

from abc import abstractmethod
from typing import Optional, Tuple, Union

from minitorch import operators
from minitorch.autodiff.variable import BaseFunction, Context, History, Variable


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

    def __hash__(self):
        return hash((self.data, self.name))

    @property
    def data(self) -> float:
        return self._data

    @data.setter
    def data(self, value: Union[int, float]) -> None:
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

    def __lt__(self, other: Union[int, float, Scalar]) -> float:
        return LT.apply(self, other)

    def __gt__(self, other: Union[int, float, Scalar]) -> float:
        return GT.apply(self, other)

    def __eq__(self, other: Union[int, float, Scalar]) -> float:
        return EQ.apply(self, other)

    def __neg__(self) -> Scalar:
        return Neg.apply(self)

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
    def data_type(cls, value: Optional = None) -> Union[type(float), float]:
        if value is not None:
            return float(value)
        return float

    @classmethod
    def variable(cls, value, history: History = History()) -> Scalar:
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
        return cls.data_type(cls._forward(ctx, *values))

    @classmethod
    @abstractmethod
    def _forward(cls, ctx: Context, *values) -> float:
        ...

    @classmethod
    def backward(cls, ctx: Context, d_out: float) -> float:
        """
        Backward call.

        Args:
            ctx - Container
                A container object that holds any information recorded during the forward call.
            d_out - float
                Derivative is multiplied by this value.
        """
        return cls.data_type(cls._backward(ctx, d_out))

    @classmethod
    @abstractmethod
    def _backward(cls, ctx: Context, d_out: float) -> float:
        ...


class Add(ScalarFunction):
    """Addition function f(x, y) = x + y"""

    @classmethod
    def _forward(cls, ctx: Context, a: float, b: float) -> float:
        return operators.add(a, b)

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, float]:
        return d_out, d_out


class Log(ScalarFunction):
    """Log function f(x) = log(x)"""

    @classmethod
    def _forward(cls, ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.log(a)

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> float:
        a = ctx.saved_values
        return operators.log_diff(a, d_out)


class Mul(ScalarFunction):
    """Multiplication for Scalars: f(x, y) = x * y"""

    @classmethod
    def _forward(cls, ctx: Context, a: float, b: float) -> float:
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, float]:
        a, b = ctx.saved_values
        return operators.mul(b, d_out), operators.mul(a, d_out)


class Inv(ScalarFunction):
    """Inverse function for Scalars: f(x) = 1(x)"""

    @classmethod
    def _forward(cls, ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.inv(a)

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> float:
        a = ctx.saved_values
        return operators.inv_diff(a, d_out)


class Neg(ScalarFunction):
    """Negation function for Scalars: f(x) = -x"""

    @classmethod
    def _forward(cls, ctx: Context, a: float) -> float:
        return operators.neg(a)

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> float:
        return operators.mul(-1, d_out)


class Sigmoid(ScalarFunction):
    """Sigmoid function applied to Scalars: f(x) = 1. / (1. + e^-x)"""

    @classmethod
    def _forward(cls, ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> float:
        a = ctx.saved_values
        return operators.sigmoid_diff(a, d_out)


class ReLU(ScalarFunction):
    """ReLU function applied to Scalars: f(x) = relu(x)"""

    @classmethod
    def _forward(cls, ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.relu(a)

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> float:
        a = ctx.saved_values
        return operators.relu_diff(a, d_out)


class Exp(ScalarFunction):
    """exp function applied to Scalars: f(x) = exp(x)"""

    @classmethod
    def _forward(cls, ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.exp(a)

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> float:
        a = ctx.saved_values
        return d_out * operators.exp(a)


class LT(ScalarFunction):
    """Less than function on scalars: f(x, y) = 1.0 if x < y else 0."""

    @classmethod
    def _forward(cls, ctx: Context, a: float, b: float) -> float:
        return operators.lt(a, b)

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> float:
        return 0.0


class GT(ScalarFunction):
    """Greater than function for scalars: f(x, y) = 1. if x > y else 0."""

    @classmethod
    def _forward(cls, ctx: Context, a: float, b: float) -> float:
        return operators.gt(a, b)

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> float:
        return 0.0


class EQ(ScalarFunction):
    """Equality function on scalars: f(x, y) = 1. if x == y else 0."""

    @classmethod
    def _forward(cls, ctx: Context, a: float, b: float) -> float:
        return operators.eq(a, b)

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> float:
        return 0.0
