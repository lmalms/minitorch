from hypothesis import given

from strategies import small_floats
from minitorch.autodiff import Scalar
from minitorch.operators import is_close, relu


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
def test_relu(x: float, y: float) -> None:
    z = Scalar(x).relu() + Scalar(y).relu()
    assert is_close(z.data, relu(x) + relu(y))

