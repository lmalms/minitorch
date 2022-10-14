from typing import Callable, Iterable, List, Tuple

import pytest
from hypothesis import given
from hypothesis.strategies import DataObject, data, lists, permutations

from minitorch.autodiff import Tensor, tensor
from minitorch.operators import is_close
from minitorch.testing import MathTestVariable

from .strategies import small_floats
from .tensor_strategies import indices, shaped_tensors, tensors

one_arg, two_arg, red_arg = MathTestVariable._comp_testing()


@given(lists(small_floats, min_size=1))
def test_create_and_index(data: List[float]) -> None:
    """
    Test 1D tensor creation and indexing
    """
    t = tensor(data)
    for idx in range(len(data)):
        assert data[idx] == t[idx]


@given(tensors())
@pytest.mark.parametrize("fn", one_arg)
def test_one_arg(
    fn: Tuple[str, Callable[[float], float], Callable[[Tensor], Tensor]], t: Tensor
) -> None:
    """Test one arg functions and compare to floats."""
    _, base_fn, tensor_fn = fn
    t_out = tensor_fn(t)
    for idx in t_out.data.indices():
        assert is_close(t_out[idx], base_fn(t[idx]))


@given(shaped_tensors(2))
@pytest.mark.parametrize("fn", two_arg)
def test_two_arg(
    fn: Tuple[str, Callable[[float, float], float], Callable[[Tensor, Tensor], Tensor]],
    ts: Tuple[Tensor, Tensor],
) -> None:
    """
    Test two arg forward funcs and compare to float implementations.
    """
    _, base_fn, tensor_fn = fn
    t1, t2 = ts
    t_out = tensor_fn(t1, t2)
    for idx in t_out.data.indices():
        assert is_close(t_out[idx], base_fn(t1[idx], t2[idx]))
