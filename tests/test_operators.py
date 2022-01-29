from hypothesis import given
from hypothesis.strategies import lists
import pytest

from minitorch.operators import (
    mul,
    id,
    add,
    neg,
    lt,
    eq,
    max,
    is_close,
    sigmoid,
    relu,
    log,
    exp,
    log_back,
    inv,
    inv_back,
    relu_back,
)
from tests.strategies import small_floats, assert_close


# Hypothesis tests for basic operators
@given(small_floats, small_floats)
def test_same_as_python(x: float, y: float) -> None:
    """Check that basic operators are the same as python implementation."""
    assert_close(mul(x, y), x * y)
    assert_close(add(x, y), x + y)
    assert_close(neg(x), -x)
    assert_close(max(x, y), x if x > y else y)
    if x != 0.:
        assert_close(inv(x), 1./x)


@given(small_floats)
def test_relu(x: float) -> None:
    if x > 0.:
        assert relu(x) == x
    else:
        assert relu(x) == 0.


@given(small_floats, small_floats)
def test_relu_back(x: float, y: float) -> None:
    if x > 0.0:
        assert relu_back(x, y) == y
    else:
        assert relu_back(x, y) == 0.


