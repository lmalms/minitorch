import random
from typing import Any, Callable, List, Tuple

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.strategies import DataObject, data, integers, lists, permutations

from minitorch.autodiff import (
    FastOps,
    SimpleOps,
    Tensor,
    TensorBackend,
    grad_check,
    rand,
    tensor,
)
from minitorch.autodiff.tensor_data import Shape
from minitorch.operators import is_close
from minitorch.testing import MathTestTensor

from .strategies import small_floats
from .tensor_strategies import shaped_tensors, tensors

# Define functions to test
one_arg, two_arg, red_arg = MathTestTensor._comp_testing()

# Define tensor backends
BACKENDS = {"simple": TensorBackend(SimpleOps), "fast": TensorBackend(FastOps)}


@given(lists(small_floats, min_size=1))
@pytest.mark.parametrize("backend", (pytest.param("simple"), pytest.param("fast")))
def test_create_and_index(backend: str, data: List[float]) -> None:
    """
    Test 1D tensor creation and indexing
    """
    t = tensor(data, backend=BACKENDS[backend])
    for idx in range(len(data)):
        assert data[idx] == t[idx]


@pytest.mark.xfail
def test_index_fail() -> None:
    t = tensor([[2, 3, 4], [4, 5, 7]])
    assert t.shape == (2, 3)
    t[50, 2]


def test_from_list() -> None:
    """
    Test longer from list conversion.
    """
    t = tensor([[2, 3, 4], [4, 5, 7]])
    assert t.shape == (2, 3)

    t = tensor([[[2, 3, 4], [4, 5, 7]]])
    assert t.shape == (1, 2, 3)


def test_from_numpy() -> None:
    t = tensor([[2, 3, 4], [4, 5, 7]])
    assert t.shape == (2, 3)
    t_to_np = t.to_numpy()
    t_from_numpy = tensor(t_to_np.tolist())
    for idx in t.data.indices():
        assert t[idx] == t_from_numpy[idx]


@pytest.mark.parametrize("backend", (pytest.param("simple"), pytest.param("fast")))
def test_view(backend: str) -> None:
    """
    Test view.
    """
    t = tensor([[2, 3, 4], [4, 5, 7]], backend=BACKENDS[backend])
    assert t.shape == (2, 3)

    t_view = t.view(6)
    assert t_view.shape == (6,)

    t_view = t_view.view(1, 6)
    assert t_view.shape == (1, 6)

    t_view = t_view.view(6, 1)
    assert t_view.shape == (6, 1)

    t_view = t_view.view(2, 3)
    assert t.is_close(t_view).all().item() == 1.0


@given(data())
@settings(max_examples=100)
@pytest.mark.parametrize("backend", (pytest.param("simple"), pytest.param("fast")))
def test_view_grad(backend: str, data: DataObject) -> None:
    """
    Test the gradient of view.
    """

    def view(x: Tensor) -> Tensor:
        x = x.contiguous()
        return x.view(x.size)

    t = data.draw(tensors(backend=BACKENDS[backend]))
    grad_check(view, t)


@pytest.mark.xfail
@pytest.mark.parametrize("backend", (pytest.param("simple"), pytest.param("fast")))
def test_permute_view_fail(backend: str) -> None:
    """
    Tensors have to be contiguous to view.
    """
    t = tensor([[2, 3, 4], [4, 5, 7]], backend=BACKENDS[backend])
    assert t.shape == (2, 3)
    t_permute = t.permute(1, 0)
    t_permute.view(6)


@given(data())
@settings(max_examples=100)
@pytest.mark.parametrize("backend", (pytest.param("simple"), pytest.param("fast")))
def test_permute_grad(backend: str, data: DataObject) -> None:
    """
    Tests permute function
    """
    t = data.draw(tensors(backend=BACKENDS[backend]))
    permutation = data.draw(permutations(range(len(t.shape))))

    def permute(x: Tensor) -> Tensor:
        return x.permute(*permutation)

    grad_check(permute, t)


@given(data())
@settings(max_examples=100)
@pytest.mark.parametrize("fn", one_arg)
@pytest.mark.parametrize("backend", (pytest.param("simple"), pytest.param("fast")))
def test_one_arg_forward(
    fn: Tuple[str, Callable[[float], float], Callable[[Tensor], Tensor]],
    backend: str,
    data: DataObject,
) -> None:
    """Test one arg forward functions and compare to floats."""
    t = data.draw(tensors(backend=BACKENDS[backend]))
    _, base_fn, tensor_fn = fn
    t_out = tensor_fn(t)
    for idx in t_out.data.indices():
        assert is_close(t_out[idx], base_fn(t[idx]))


@given(data())
@settings(max_examples=100)
@pytest.mark.parametrize("fn", two_arg)
@pytest.mark.parametrize("backend", (pytest.param("simple"), pytest.param("fast")))
def test_two_arg_forward(
    fn: Tuple[str, Callable[[float, float], float], Callable[[Tensor, Tensor], Tensor]],
    backend: str,
    data: DataObject,
) -> None:
    """
    Test two arg forward funcs and compare to float implementations.
    """
    t1, t2 = data.draw(shaped_tensors(2, backend=BACKENDS[backend]))
    _, base_fn, tensor_fn = fn
    t_out = tensor_fn(t1, t2)
    for idx in t_out.data.indices():
        assert is_close(t_out[idx], base_fn(t1[idx], t2[idx]))


@pytest.mark.parametrize(
    ["td_in", "dim", "td_out", "out_shape"],
    [
        ([[[1, 2, 3], [4, 5, 6]]], 0, [[[1, 2, 3], [4, 5, 6]]], (1, 2, 3)),
        ([[[1, 2, 3], [4, 5, 6]]], 1, [[[5, 7, 9]]], (1, 1, 3)),
        ([[[1, 2, 3], [4, 5, 6]]], 2, [[[6], [15]]], (1, 2, 1)),
        ([[[1, 2, 3], [4, 5, 6]]], None, [21], (1,)),
    ],
)
@pytest.mark.parametrize("backend", (pytest.param("simple"), pytest.param("fast")))
def test_reduce_sum(
    td_in: Any,
    dim: int,
    td_out: Any,
    out_shape: Shape,
    backend: str,
) -> None:
    t_in = tensor(td_in, backend=BACKENDS[backend])
    t_out = tensor(td_out, backend=BACKENDS[backend])
    assert t_in.shape == (1, 2, 3)

    t_summed = t_in.sum(dim=dim)
    assert t_summed.shape == out_shape
    assert t_summed.is_close(t_out).all().item() == 1.0


@given(data())
@pytest.mark.parametrize("fn", one_arg)
@pytest.mark.parametrize("backend", (pytest.param("simple"), pytest.param("fast")))
def test_one_arg_grad(
    fn: Tuple[str, Callable[[float], float], Callable[[Tensor], Tensor]],
    backend: str,
    data: DataObject,
) -> None:
    """
    Test autograd on one-arg funcs and compare using central difference.
    """
    t = data.draw(tensors(backend=BACKENDS[backend]))
    _, _, tensor_fn = fn
    grad_check(tensor_fn, t)


@given(data())
@settings(max_examples=100)
@pytest.mark.parametrize("fn", two_arg)
@pytest.mark.parametrize("backend", (pytest.param("simple"), pytest.param("fast")))
def test_two_arg_grad(
    fn: Tuple[str, Callable[[float], float], Callable[[Tensor], Tensor]],
    backend: str,
    data: DataObject,
) -> None:
    """
    Test the grads of two arg tensor functions.
    """
    tensors = data.draw(shaped_tensors(2, backend=BACKENDS[backend]))
    _, _, tensor_fn = fn
    grad_check(tensor_fn, *tensors)


@given(data())
@settings(max_examples=100)
@pytest.mark.parametrize("fn", two_arg)
@pytest.mark.parametrize("backend", (pytest.param("simple"), pytest.param("fast")))
def test_two_arg_grad_broadcast(
    fn: Tuple[str, Callable[[float, float], float], Callable[[Tensor, Tensor], Tensor]],
    backend: str,
    data: DataObject,
) -> None:
    t1, t2 = data.draw(shaped_tensors(2, backend=BACKENDS[backend]))
    _, _, tensor_fn = fn
    grad_check(tensor_fn, t1, t2)

    # Broadcast check
    grad_check(tensor_fn, t1.sum(0), t2)
    grad_check(tensor_fn, t1, t2.sum(0))


@given(data())
@settings(max_examples=100)
@pytest.mark.parametrize("fn", red_arg)
@pytest.mark.parametrize("backend", (pytest.param("simple"), pytest.param("fast")))
def test_reduce_grad(
    fn: Tuple[str, Callable[[List[float]], float], Callable[[Tensor], Tensor]],
    backend: str,
    data: DataObject,
):
    t = data.draw(tensors(backend=BACKENDS[backend]))
    _, _, tensor_fn = fn
    grad_check(tensor_fn, t)


def test_grad_size() -> None:
    """
    Test size of gradients.
    """
    x = tensor([1], requires_grad=True)
    y = tensor([[1, 1]], requires_grad=True)
    z = (x * y).sum()
    z.backward()

    assert z.shape == (1,)
    assert x.grad is not None
    assert y.grad is not None
    assert x.shape == x.grad.shape
    assert y.grad.shape == y.grad.shape


@given(data())
def test_mat_mul(data: DataObject) -> None:
    small_ints = integers(min_value=2, max_value=4)
    W, X, Y, Z = (
        data.draw(small_ints),
        data.draw(small_ints),
        data.draw(small_ints),
        data.draw(small_ints),
    )
    x = data.draw(tensors(backend=BACKENDS["fast"], shape=(Z, W, X)))
    y = data.draw(tensors(backend=BACKENDS["fast"], shape=(1, X, Y)))
    z_mat_mul = x @ y
    z_zip = x.contiguous().view(Z, W, X, 1) * y.contiguous().view(1, 1, X, Y)
    z_zip_red = z_zip.sum(dim=2).view(Z, W, Y)

    assert np.allclose(z_mat_mul.data.storage, z_zip_red.data.storage)


def test_mat_mul_grad() -> None:
    min_dim, max_dim = 1, 10
    for _ in range(10):
        n_rows_x = random.randint(min_dim, max_dim)
        n_cols_y = random.randint(min_dim, max_dim)
        n_reduce = random.randint(min_dim, max_dim)

        x = rand((n_rows_x, n_reduce), backend=BACKENDS["fast"])
        y = rand((n_reduce, n_cols_y), backend=BACKENDS["fast"])
        z_mat_mul = x @ y

        # Compare to zip-reduce implementation
        z_zip = x.view(*x.shape, 1) * y.view(1, *y.shape)
        z_zip_red = z_zip.sum(dim=1).view(n_rows_x, n_cols_y)

        for idx in z_zip_red.data.indices():
            assert is_close(z_zip_red[idx], z_mat_mul[idx])

        # Check grad
        def mat_mul(x: Tensor, y: Tensor):
            return x @ y

        grad_check(mat_mul, x, y)
