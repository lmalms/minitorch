from typing import Optional
from abc import abstractmethod

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
        self.data = float(value)

    def __repr__(self) -> str:
        return f"Scalar({self.data})"

    def __mul__(self, other):
        pass


class ScalarFunction(BaseFunction):
    """
    Wrapper for a mathematical function that processes and produces Scalar variables.
    """
    pass


class Add(ScalarFunction):
    """Addition function f(x, y) = x + y"""

    @staticmethod
    def forward(cls, ctx: Context, a, b):
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output):
        return d_output, d_output