from typing import Tuple

from minitorch import operators
from minitorch.autodiff import Context, Scalar, ScalarFunction


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


def test_backprop1():
    """
    Example 1: F1(0.0, var1)
    """
    var1 = Scalar(0.0)
    var2 = Function1.apply(0.0, var1)
    var2.backward(d_out=5.0)
    assert var1.derivative == 5.0


def test_backprop2():
    """
    Example 2: F1(0.0, F1(0.0, var1))
    """
    var1 = Scalar(0.0)
    var2 = Function1.apply(0.0, var1)
    var3 = Function1.apply(0.0, var2)
    var3.backward(d_out=5.0)
    assert var1.derivative == 5.0


def test_backprop3():
    """
    Example 3: F1(F1(0.0, var1), F1(0.0, var1))
    """
    var1 = Scalar(0.0)
    var2 = Function1.apply(0.0, var1)
    var3 = Function1.apply(0.0, var1)
    var4 = Function1.apply(var2, var3)
    var4.backward(d_out=5.0)
    assert var1.derivative == 10.0


def test_backprop4():
    """
    Example 4: F1(F1(0.0, F1(0.0, var0), F1(0.0, F1(0.0, var0))
    """
    var0 = Scalar(0.0)
    var1 = Function1.apply(0.0, var0)
    var2 = Function1.apply(0.0, var1)
    var3 = Function1.apply(0.0, var1)
    var4 = Function1.apply(var2, var3)
    var4.backward(d_out=5.0)
    assert var0.derivative == 10.0
