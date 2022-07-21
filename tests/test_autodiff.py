from typing import Tuple

import pytest

from minitorch import operators
from minitorch.autodiff import Context, History, Scalar, ScalarFunction, Variable


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
        return operators.mul(d_out, operators.add(y, 1)), operators.mul(d_out, x)


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


def test_chain_rule3():
    """Check that constants are ignored but variables get derivvatives."""
    const = 10
    var = Scalar(value=5.0)
    ctx = Context(requires_grad_=True)
    _ = Function2.forward(ctx, const, var.data)

    back = Function2.chain_rule(ctx=ctx, inputs=[const, var], d_out=5)
    assert len(back) == 1
    variable, derivative = back[0]
    assert variable.name == var.name
    assert derivative == operators.mul(5, 10)


def test_chain_rule4():
    """Check that derivatives are computed w.r.t. all variables."""
    var1 = Scalar(5.0)
    var2 = Scalar(10.0)
    ctx = Context(requires_grad_=True)
    _ = Function2.forward(ctx, var1.data, var2.data)

    back = Function2.chain_rule(ctx=ctx, inputs=[var1, var2], d_out=5)
    assert len(back) == 2

    # Check first derivative
    variable, derivative = back[0]
    assert variable.name == var1.name
    assert derivative == 5 * (10 + 1)

    # Check second derivative
    variable, derivative = back[1]
    assert variable.name == var2.name
    assert derivative == 5 * 5
