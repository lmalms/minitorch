from hypothesis import given

from strategies import small_floats
from minitorch.autodiff import Scalar
from minitorch.operators import is_close, relu


EPS = 1e-09


@given(small_floats, small_floats)
def test_add(x: float, y: float) -> None:
    z = Scalar(x) + Scalar(y)
    assert is_close(z.data, x + y)

    z = Scalar(y) + Scalar(x)
    assert is_close(z.data, x + y)

    z = Scalar(x) + y
    assert is_close(z.data, x + y)

    z = x + Scalar(y)
    assert is_close(z.data, x + y)


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


@given(small_floats, small_floats)
def test_log(x: float, y: float) -> None:
    pass


@given(small_floats, small_floats)
def test_exp(x: float, y: float) -> None:
    pass


@given(small_floats, small_floats)
def test_sigmoid(x: float, y: float) -> None:
    pass


@given(small_floats, small_floats)
def test_relu(x: float, y: float) -> None:
    z = Scalar(x).relu() + Scalar(y).relu()
    assert is_close(z.data, relu(x) + relu(y))
