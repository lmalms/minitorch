import pytest
from hypothesis import given

from minitorch.autodiff import Scalar, derivative_check
from minitorch.operators import add, exp, inv, is_close, log, mul, relu, sigmoid
from minitorch.testing import MathTestScalars
from tests.strategies import (
    small_floats,
    small_positive_floats,
    small_scalars,
    tiny_floats,
)

# Test scalar operators


@given(small_floats, small_floats)
def test_add(x: float, y: float) -> None:
    z = Scalar(x) + Scalar(y)
    assert z.data == (x + y)

    z = Scalar(y) + Scalar(x)
    assert z.data == (x + y)

    z = Scalar(x) + y
    assert z.data == (x + y)

    z = x + Scalar(y)
    assert z.data == (x + y)


@given(small_floats, small_floats)
def test_mul(x: float, y: float) -> None:
    z = Scalar(x) * Scalar(y)
    assert is_close(z.data, x * y)

    z = Scalar(y) * Scalar(x)
    assert is_close(z.data, x * y)

    z = Scalar(x) * y
    assert is_close(z.data, x * y)

    z = x * Scalar(y)
    assert is_close(z.data, x * y)


@given(small_floats, small_floats)
def test_div(x: float, y: float) -> None:
    z = Scalar(x) / Scalar(y)
    assert is_close(z.data, mul(x, inv(y)))

    z = x / Scalar(y)
    assert is_close(z.data, mul(x, inv(y)))

    z = Scalar(x) / y
    assert is_close(z.data, mul(x, inv(y)))


@given(small_floats)
def test_lt(x: float) -> None:
    assert (Scalar(x - 1) < Scalar(x)) == 1.0
    assert (Scalar(x) < Scalar(x - 1)) == 0.0

    assert (Scalar(x - 1) < x) == 1.0
    assert (Scalar(x) < (x - 1)) == 0.0

    assert ((x - 1) < Scalar(x)) == 1.0
    assert (x < Scalar(x - 1)) == 0.0

    assert (Scalar(x) < Scalar(x)) == 0.0
    assert (x < Scalar(x)) == 0.0
    assert (Scalar(x) < x) == 0.0


@given(small_floats)
def test_gt(x: float) -> None:
    assert (Scalar(x) > Scalar(x - 1)) == 1.0
    assert (Scalar(x - 1) > Scalar(x)) == 0.0

    assert (Scalar(x) > (x - 1)) == 1.0
    assert (Scalar(x - 1) > x) == 0.0

    assert (x > Scalar(x - 1)) == 1.0
    assert ((x - 1) > Scalar(x)) == 0.0

    assert (Scalar(x) > Scalar(x)) == 0.0
    assert (x > Scalar(x)) == 0.0
    assert (Scalar(x) > x) == 0.0


@given(small_floats)
def test_eq(x: float) -> None:
    assert (Scalar(x) == Scalar(x)) == 1.0
    assert (Scalar(x - 1) == Scalar(x)) == 0.0
    assert (Scalar(x) == x) == 1.0
    assert (x == Scalar(x)) == 1.0


@given(small_floats)
def test_le(x: float) -> None:
    assert (Scalar(x - 1) <= Scalar(x)) == 1.0
    assert (Scalar(x) <= Scalar(x - 1)) == 0.0

    assert (Scalar(x - 1) <= x) == 1.0
    assert (Scalar(x) <= (x - 1)) == 0.0

    assert ((x - 1) <= Scalar(x)) == 1.0
    assert (x <= Scalar(x - 1)) == 0.0

    assert (Scalar(x) <= Scalar(x)) == 1.0
    assert (x <= Scalar(x)) == 1.0
    assert (Scalar(x) <= x) == 1.0


@given(small_floats)
def test_ge(x: float) -> None:
    assert (Scalar(x) >= Scalar(x - 1)) == 1.0
    assert (Scalar(x - 1) >= Scalar(x)) == 0.0

    assert (Scalar(x) >= (x - 1)) == 1.0
    assert (Scalar(x - 1) >= x) == 0.0

    assert (x >= Scalar(x - 1)) == 1.0
    assert ((x - 1) > Scalar(x)) == 0.0

    assert (Scalar(x) >= Scalar(x)) == 1.0
    assert (x >= Scalar(x)) == 1.0
    assert (Scalar(x) >= x) == 1.0


@given(small_floats)
def test_square(x: float) -> None:
    square_scalar = Scalar(x).square()
    assert is_close(square_scalar.data, x**2)

    z = Scalar(x).square() + Scalar(x).square()
    assert is_close(z.data, x**2 + x**2)


@given(tiny_floats)
def test_cube(x: float) -> None:
    cube_scalar = Scalar(x).cube()
    assert is_close(cube_scalar.data, x**3)

    z = Scalar(x).cube() + Scalar(x).cube()
    assert is_close(z.data, x**3 + x**3)


@given(small_floats)
def test_neg(x: float) -> None:
    neg_scalar = -Scalar(x)
    assert neg_scalar.data == -x

    assert -Scalar(x) == -x


@given(small_positive_floats, small_positive_floats)
def test_log(x: float, y: float) -> None:
    log_scalar = Scalar(x).log()
    assert is_close(log_scalar.data, log(x))

    z = Scalar(x).log() + Scalar(y).log()
    assert is_close(z.data, add(log(x), log(y)))


@given(tiny_floats, tiny_floats)
def test_exp(x: float, y: float) -> None:
    exp_scalar = Scalar(x).exp()
    assert is_close(exp_scalar.data, exp(x))

    z = Scalar(x).exp() + Scalar(y).exp()
    assert is_close(z.data, add(exp(x), exp(y)))


@given(small_floats, small_floats)
def test_sigmoid(x: float, y: float) -> None:
    sigmoid_scalar = Scalar(x).sigmoid()
    assert is_close(sigmoid_scalar.data, sigmoid(x))

    z = Scalar(x).sigmoid() + Scalar(y).sigmoid()
    assert is_close(z.data, add(sigmoid(x), sigmoid(y)))


@given(small_floats, small_floats)
def test_relu(x: float, y: float) -> None:
    relu_scalar = Scalar(x).relu()
    assert is_close(relu_scalar.data, relu(x))

    z = Scalar(x).relu() + Scalar(y).relu()
    assert is_close(z.data, relu(x) + relu(y))


# One and two argument functions with scalars

one_arg_funcs, two_arg_funcs, _ = MathTestScalars._comp_testing()


@given(small_scalars)
@pytest.mark.parametrize("fn", one_arg_funcs)
def test_one_arg_derivative(fn, x: Scalar):
    _, _, scalar_fn = fn
    derivative_check(scalar_fn, x)


@given(small_scalars, small_scalars)
@pytest.mark.parametrize("fn", two_arg_funcs)
def test_two_arg_derivative(fn, x: Scalar, y: Scalar):
    _, _, scalar_fn = fn
    derivative_check(scalar_fn, x, y)


def test_scalar_name():
    x = Scalar(10, name="x")
    y = (x + 10.0) * 20
    y.name = "y"
    return y
