from hypothesis import given

from strategies import small_floats, tiny_floats, small_positive_floats
from minitorch.autodiff import Scalar
from minitorch.operators import is_close, relu, add, log, exp, sigmoid


EPS = 1e-09


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
    z = Scalar(x) / (Scalar(y) + EPS)
    assert is_close(z.data, x / (y + EPS))

    z = x / (Scalar(y) + EPS)
    assert is_close(z.data, x / (y + EPS))

    z = Scalar(x) / (y + EPS)
    assert is_close(z.data, x / (y + EPS))


@given(small_floats)
def test_lt(x: float) -> None:
    assert Scalar(x - 1) < Scalar(x) == 1.0
    assert Scalar(x) < Scalar(x - 1) == 0.0

    assert Scalar(x - 1) < x == 1.0
    assert Scalar(x) < (x - 1) == 0.0

    assert (x - 1) < Scalar(x) == 1.0
    assert x < Scalar(x - 1) == 0.0

    assert Scalar(x) < Scalar(x) == 0.0
    assert x < Scalar(x) == 0.0
    assert Scalar(x) < x == 0.0


@given(small_floats, small_floats)
def test_gt(x: float) -> None:
    assert Scalar(x) > Scalar(x - 1) == 1.0
    assert Scalar(x - 1) > Scalar(x) == 0.0

    assert Scalar(x) > (x - 1) == 1.0
    assert Scalar(x - 1) > x == 0.0

    assert x > Scalar(x - 1) == 1.0
    assert (x - 1) > Scalar(x) == 0.0

    assert Scalar(x) > Scalar(x) == 0.0
    assert x > Scalar(x) == 0.0
    assert Scalar(x) > x == 0.0


@given(small_floats)
def test_eq(x: float) -> None:
    assert (Scalar(x) == Scalar(x)) == 1.0
    assert (Scalar(x - 1) == Scalar(x)) == 0.0
    assert (Scalar(x) == x) == 1.0
    assert (x == Scalar(x)) == 1.0


@given(small_floats)
def test_neg(x: float) -> None:
    neg_scalar = -Scalar(x)
    assert neg_scalar.data == - x

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
