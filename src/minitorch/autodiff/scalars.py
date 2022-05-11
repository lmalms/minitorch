from abc import abstractmethod
from typing import Optional, Union, Type, Tuple

from minitorch import operators
from minitorch.autodiff.variable import History, Context, Variable, BaseFunction


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
            self,
            value: float,
            history: History = History(),
            name: Optional[str] = None
    ):
        super().__init__(history=history, name=name)
        self.data = value

    @property
    def data(self) -> float:
        return self._data

    @data.setter
    def data(self, value: float) -> None:
        if not isinstance(value, float):
            raise TypeError(f"Scalar values have to be of type float - got {type(value)}.")
        self._data = value

    def __repr__(self) -> str:
        return f"Scalar({self.data})"

    def __mul__(self, other):
        pass


class ScalarFunction(BaseFunction):
    """
    Wrapper for a mathematical function that processes and produces Scalar variables.
    """

    @classmethod
    def data_type(cls, value: Optional = None) -> Union[Type[float], float]:
        if value is None:
            return float
        else:
            return float(value)

    @classmethod
    def variable(cls, value, history: History = History()) -> Scalar:
        return Scalar(value, history)

    @classmethod
    @abstractmethod
    def forward(cls, ctx: Context, *values):
        """
        Forward call.
        Args:
            ctx - Context
                A context container to save any information to that may be needed for
                backward call.
            *values - List[float]
                n floats to run forward call over.
        """
        ...

    @classmethod
    @abstractmethod
    def backward(cls, ctx: Context, d_out: float) -> float:
        """
        Backward call.

        Args:
            ctx - Container
                A container object that holds any information recorded duing the forward call.
            d_out - float
                Derivative is multiplied by this value.
        """
        ...


class Add(ScalarFunction):
    """Addition function f(x, y) = x + y"""

    @classmethod
    def forward(cls, ctx: Context, a: float, b: float) -> float:
        return a + b

    @classmethod
    def backward(cls, ctx: Context, d_out: float) -> Tuple[float, float]:
        return d_out, d_out


class Log(ScalarFunction):
    """Log function f(x) = log(x)"""

    @classmethod
    def forward(cls, ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.log(a)

    @classmethod
    def backward(cls, ctx: Context, d_out: float) -> float:
        a = ctx.saved_values
        return operators.log_diff(a, d_out)


class Mul(ScalarFunction):
    """Multiplication for Scalars: f(x, y) = x * y"""

    @classmethod
    def forward(cls, ctx: Context, a: float, b: float) -> float:
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @classmethod
    def backward(cls, ctx: Context, d_out: float) -> Tuple[float, float]:
        a, b = ctx.saved_values
        return operators.mul(b, d_out), operators.mul(a, d_out)

