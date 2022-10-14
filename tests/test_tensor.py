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


def test_from_list() -> None:
    """
    Test longer from list conversion.
    """
    t = tensor([[2, 3, 4], [4, 5, 7]])
    assert t.shape == (2, 3)

    t = tensor([[[2, 3, 4], [4, 5, 7]]])
    assert t.shape == (1, 2, 3)


def test_view() -> None:
    """
    Test view.
    """
    t = tensor([[2, 3, 4], [4, 5, 7]])
    assert t.shape == (2, 3)

    t_view = t.view(6)
    assert t_view.shape == (6,)

    t_view = t_view.view(1, 6)
    assert t_view.shape == (1, 6)

    t_view = t_view.view(6, 1)
    assert t_view.shape == (6, 1)

    t_view = t_view.view(2, 3)
    assert t.is_close(t_view).all().item() == 1.0


@pytest.mark.xfail
def test_permute_view_fail() -> None:
    """
    Tensors have to be contiguous to view.
    """
    t = tensor([[2, 3, 4], [4, 5, 7]])
    assert t.shape == (2, 3)
    t_permute = t.permute(1, 0)
    t_permute.view(6)


@pytest.mark.xfail
def test_index_fail() -> None:
    t = tensor([[2, 3, 4], [4, 5, 7]])
    assert t.shape == (2, 3)
    t[50, 2]


def test_from_numpy() -> None:
    t = tensor([[2, 3, 4], [4, 5, 7]])
    assert t.shape == (2, 3)
    t_to_np = t.to_numpy()
    t_from_numpy = tensor(t_to_np.tolist())
    for idx in t.data.indices():
        assert t[idx] == t_from_numpy[idx]


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
