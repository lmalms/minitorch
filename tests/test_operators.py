from hypothesis import given
from hypothesis.strategies import lists
import pytest

from minitorch.operators import (
    mul,
    id,
    add,
    neg,
    lt,
    gt,
    eq,
    max,
    is_close,
    sigmoid,
    relu,
    log,
    exp,
    log_diff,
    inv,
    inv_diff,
    relu_diff,
)
from tests.strategies import (
    assert_close,
    small_floats,
    small_positive_floats,
    tiny_floats,
    tiny_positive_floats
)


# Hypothesis tests for basic operators
@given(small_floats, small_floats)
def test_same_as_python(x: float, y: float) -> None:
    """Check that basic operators are the same as python implementation."""
    assert mul(x, y) == x * y
    assert add(x, y) == x + y
    assert neg(x) == -x
    assert max(x, y) == (x if x > y else y)
    if x != 0.:
        assert_close(inv(x), 1./x)


@given(small_floats)
def test_id(x: float) -> None:
    assert id(x) == x


@given(small_floats)
def test_neg(x: float) -> None:
    assert neg(-x) == x


@given(small_floats)
def test_lt(x: float) -> None:
    assert lt(x - 1., x) == 1.
    assert lt(x, x - 1.) == 0.


@given(small_floats)
def test_gt(x: float) -> None:
    assert gt(x + 1., x) == 1.
    assert gt(x, x + 1.) == 0.


@given(small_floats)
def test_max(x: float) -> None:
    assert max(x - 1., x) == x
    assert max(x, x - 1.) == x
    assert max(x + 1., x) == x + 1.
    assert max(x, x + 1.) == x + 1.


@given(small_floats)
def test_eq(x: float) -> None:
    assert eq(x, x) == 1.
    assert eq(x, x - 1.) == 0.
    assert eq(x, x + 1.) == 0.
    assert eq(x + 1., x + 1.) == 1.


@given(small_floats)
def test_relu(x: float) -> None:
    if x > 0.:
        assert relu(x) == x
    else:
        assert relu(x) == 0.


@given(small_floats, small_floats)
def test_relu_back(x: float, y: float) -> None:
    if x > 0.0:
        assert relu_diff(x, y) == y
    else:
        assert relu_diff(x, y) == 0.


@given(small_floats)
def test_sigmoid(x: float) -> None:
    """
    Should always have a value between 0. and 1.
    1 - sigmoid(x) = sigmoid(-x)
    sigmoid(0) = 0.5
    strictly increasing
    """
    assert (sigmoid(x) >= 0.) and (sigmoid(x) <= 1.)
    assert is_close(1 - sigmoid(x), sigmoid(-x))
    assert is_close(sigmoid(0.), 0.5)
    assert all(
        sigmoid(j) >= sigmoid(i) for (i, j) in zip(
            [x + k * 0.01 for k in range(10000)],
            [x + k * 0.01 for k in range(10000)][1:]
        )
    )


@given(small_floats, small_floats, small_floats)
def test_transitive(x: float, y: float, z: float) -> None:
    """
    If x < y and y < z, then x < z.
    If x > y and y > z, then x > z.
    """
    if lt(x, y) and lt(y, z):
        assert lt(x, z)

    if gt(x, y) and gt(y, z):
        assert gt(x, z)


@given(small_floats, small_floats)
def test_symmetric(x: float, y: float) -> None:
    """
    Multiplication and addition is symmetric.
    """
    assert mul(x, y) == mul(y, x)
    assert add(x, y) == add(y, x)


@given(small_floats, small_floats, small_floats)
def test_distributive(x: float, y: float, z: float) -> None:
    """
    Multiplication and addition is distributive.
    """
    assert is_close(mul(x, add(y, z)), add(mul(x, y), mul(x, z)))


@given(small_floats, small_floats, small_floats)
def test_associative(x: float, y: float, z: float):
    """
    Multiplication and addition are associative
    """
    assert is_close(add(add(x, y), z), add(x, add(y, z)))
    assert is_close(mul(mul(x, y), z), mul(x, mul(y, z)))


@given(tiny_positive_floats, tiny_positive_floats)
def test_log(x: float, y: float) -> None:
    """
    Log(1) = 0.
    Change of base -> log_a(x) = log_b(x) / log_b(a)
    Multiplication -> log_a(x * y) = log_a(x) + log_a(y)
    Division -> log_a(x / y) = log_a(x) - log_a(y)
    Exponentiation -> log_a(x)^y = y * log_a(x)
    Monotonically increasing
    """
    assert log(1.) == 0.
    assert is_close(log(x, y), (log(x) / log(y)))
    assert is_close(log(mul(x, y)), add(log(x), log(y)))
    assert is_close(log(x / y), log(x) - log(y))
    assert is_close(log(x**y), mul(y, log(x)))
    assert all(
        log(j) >= log(i) for (i, j) in zip(
            [x + k * 0.01 for k in range(1000)],
            [x + k * 0.01 for k in range(1000)][1:],
        )
    )


@given(tiny_floats, tiny_floats)
def test_exp(x: float, y: float) -> None:
    """
    Zero exponent property -> exp(0) = 1
    Negative exponent property -> exp(-x) = 1/exp(x)
    Product property -> exp(x) * exp(y) = exp(5 + 4)
    Quotient property -> exp(x) / exp(y) = exp(x - y)
    Power property -> exp(x)y = exp(x*y)
    """
    assert exp(0) == 1.
    assert is_close(exp(-x), (1 / exp(x)))
    assert is_close(mul(exp(x), exp(y)), exp(add(x, y)))
    assert is_close((exp(x) / exp(y)), exp(x - y))
    assert is_close(exp(x)**y, exp(mul(x, y)))

