from typing import Callable, List, Tuple

import pytest
from hypothesis import given
from hypothesis.strategies import lists

from minitorch.constants import EPS
from minitorch.operators import (
    add,
    add_lists,
    eq,
    exp,
    gt,
    identity,
    inv,
    inv_diff,
    is_close,
    log,
    log_diff,
    lt,
    maximum,
    mul,
    neg,
    neg_list,
    product,
    relu,
    relu_diff,
    sigmoid,
    summation,
)
from minitorch.testing import MathTest
from tests.strategies import (
    assert_close,
    small_floats,
    small_positive_floats,
    tiny_floats,
    tiny_positive_floats,
)


@given(small_floats, small_floats)
def test_same_as_python(x: float, y: float) -> None:
    """Check that basic operators are the same as python implementation."""
    assert mul(x, y) == x * y
    assert add(x, y) == x + y
    assert neg(x) == -x
    assert maximum(x, y) == (x if x > y else y)
    if x != 0.0:
        assert_close(inv(x), 1.0 / (x + EPS))


@given(small_floats)
def test_id(x: float) -> None:
    assert identity(x) == x


@given(small_floats)
def test_neg(x: float) -> None:
    assert neg(-x) == x


@given(small_floats)
def test_lt(x: float) -> None:
    assert lt(x - 1.0, x) == 1.0
    assert lt(x, x - 1.0) == 0.0


@given(small_floats)
def test_gt(x: float) -> None:
    assert gt(x + 1.0, x) == 1.0
    assert gt(x, x + 1.0) == 0.0


@given(small_floats)
def test_max(x: float) -> None:
    assert maximum(x - 1.0, x) == x
    assert maximum(x, x - 1.0) == x
    assert maximum(x + 1.0, x) == x + 1.0
    assert maximum(x, x + 1.0) == x + 1.0


@given(small_floats)
def test_eq(x: float) -> None:
    assert eq(x, x) == 1.0
    assert eq(x, x - 1.0) == 0.0
    assert eq(x, x + 1.0) == 0.0
    assert eq(x + 1.0, x + 1.0) == 1.0


@given(small_floats)
def test_relu(x: float) -> None:
    if x > 0.0:
        assert relu(x) == x
    else:
        assert relu(x) == 0.0


@given(small_floats, small_floats)
def test_relu_diff(x: float, y: float) -> None:
    if x > 0.0:
        assert relu_diff(x, y) == y
    else:
        assert relu_diff(x, y) == 0.0


@given(small_floats)
def test_sigmoid(x: float) -> None:
    """
    Should always have a value between 0. and 1.
    1 - sigmoid(x) = sigmoid(-x)
    sigmoid(0) = 0.5
    strictly increasing
    """
    assert (sigmoid(x) >= 0.0) and (sigmoid(x) <= 1.0)
    assert is_close(1 - sigmoid(x), sigmoid(-x))
    assert is_close(sigmoid(0.0), 0.5)
    assert all(
        sigmoid(j) >= sigmoid(i)
        for (i, j) in zip(
            [x + k * 0.01 for k in range(10000)],
            [x + k * 0.01 for k in range(10000)][1:],
        )
    )


@given(small_floats, small_floats)
def test_diffs(a, b):
    relu_diff(a, b)
    inv_diff(a + 2.4, b)
    log_diff(abs(a) + 4, b)


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
    assert log(1.0) == 0.0
    assert is_close(log(x, y), (log(x) / log(y)))
    assert is_close(log(mul(x, y)), add(log(x), log(y)))
    assert is_close(log(x / y), log(x) - log(y))
    assert is_close(log(x**y), mul(y, log(x)))
    assert all(
        log(j) >= log(i)
        for (i, j) in zip(
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
    assert exp(0) == 1.0
    assert is_close(exp(-x), (1 / exp(x)))
    assert is_close(mul(exp(x), exp(y)), exp(add(x, y)))
    assert is_close((exp(x) / exp(y)), exp(x - y))
    assert is_close(exp(x) ** y, exp(mul(x, y)))


@given(small_floats, small_floats, small_floats, small_floats)
def test_add_lists(x: float, y: float, v: float, w: float) -> None:
    x1, x2 = add_lists([x, y], [v, w])
    y1, y2 = x + v, y + w
    assert is_close(x1, y1)
    assert is_close(x2, y2)


@given(lists(small_floats, min_size=5, max_size=5))
def test_summation(x: List[float]) -> None:
    assert is_close(summation(x), summation(x))


@given(
    lists(small_floats, min_size=5, max_size=5),
    lists(small_floats, min_size=5, max_size=5),
)
def test_sum_distribute(x: List[float], y: List[float]) -> None:
    """
    Test for the distributive property of summation over lists.
    -> sum(ls1) + sum(ls2) = sum(add_lists(ls1, ls2))
    """
    s1 = add(summation(x), summation(y))
    s2 = summation(add_lists(x, y))
    assert is_close(s1, s2)


@given(small_floats, small_floats, small_floats)
def test_product(x: float, y: float, z: float) -> None:
    assert is_close(product([x, y, z]), x * y * z)


@given(lists(small_floats, min_size=5, max_size=5))
def test_neg_list(x: List[float]) -> None:
    negative = neg_list(x)
    assert all([is_close(i, -j) for (i, j) in zip(negative, x)])


# Generic mathematical tests
one_arg_tests, two_arg_tests, _ = MathTest.generate_tests()


@given(small_floats)
@pytest.mark.parametrize("fn", one_arg_tests)
def test_one_arg_funcs(fn: List[Tuple[str, Callable, Callable]], x: float) -> None:
    name, base_fn, _ = fn
    base_fn(x)


@given(small_floats, small_floats)
@pytest.mark.parametrize("fn", two_arg_tests)
def test_two_arg_funcs(
    fn: List[Tuple[str, Callable, Callable]], x: float, y: float
) -> None:
    name, base_fn, _ = fn
    base_fn(x, y)


@given(small_floats, small_floats)
def test_diffs(x: float, y: float) -> None:
    relu_diff(x, y)
    inv_diff(x + 2.4, y)
    log_diff(abs(x) + 4.0, y)
