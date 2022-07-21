from typing import Tuple
import pytest
from minitorch.autodiff import ScalarFunction, History, Context, Variable
from minitorch import operators


class Function1(ScalarFunction):
    @classmethod
    def forward(cls, ctx: Context, x: float, y: float) -> float:
        """f(x, y) = x + y + 10"""
        return operators.add(x, operators.add(y, 10))

    @classmethod
    def backward(cls, ctx: Context, d_out: float) -> Tuple[float, float]:
        """f'_x(x, y) = 1 ; f'_y(x, y) = 1"""
        return d_out, d_out


class Function2(ScalarFunction):
    @classmethod
    def forward(cls, ctx: Context, x: float, y: float) -> float:
        """f(x, y) = x * y + x"""
        ctx.save_for_backward(x, y)
        return operators.add(operators.mul(x, y), x)

    @classmethod
    def backward(cls, ctx: Context, d_out: float) -> Tuple[float, float]:
        """f'_x(x, y) = y + 1 ; f'_y(x, y) = x"""
        x, y = ctx.saved_values
        return operators.mul(d_out, operators.add(y, 1)), x


def test_chain_rule1():
    """Check that constants are ignored."""
    const = Variable(history=None)
    back = Function1.chain_rule(ctx=None, inputs=[const, const], d_out=5)
    assert len(list(back)) == 0


def test_chain_rule2():
    """Check that constants are ignored but variables get derivatives."""
    var = Variable(History())
    const = Variable(None)
    back = Function1.chain_rule(ctx=None, inputs=[var, const], d_out=5.0)
    assert len(back) == 1
    variable, derivative = back[0]
    assert variable.name == var.name
    assert derivative == 5.0
