from typing import Tuple

import numpy as np

import minitorch.autodiff.tensor_functions as tf
from minitorch.autodiff import TensorFunction
from minitorch import operators
from minitorch.autodiff import (
    Context,
    Scalar,
    ScalarFunction,
    Tensor,
    SimpleOps,
    TensorBackend,
    FastOps,
)
import pytest

# Define tensor backends
BACKENDS = {"simple": TensorBackend(SimpleOps), "fast": TensorBackend(FastOps)}


class ScalarFunction1(ScalarFunction):
    @classmethod
    def _forward(cls, ctx: Context, x: float, y: float) -> float:
        """f(x, y) = x + y + 10"""
        return operators.add(x, operators.add(y, 10))

    @classmethod
    def backward(cls, ctx: Context, d_out: float) -> Tuple[float, float]:
        """f'_x(x, y) = 1 ; f'_y(x, y) = 1"""
        return d_out, d_out


class ScalarFunction2(ScalarFunction):
    @classmethod
    def _forward(cls, ctx: Context, x: float, y: float) -> float:
        """f(x, y) = x * y + x"""
        ctx.save_for_backward(x, y)
        return operators.add(operators.mul(x, y), x)

    @classmethod
    def backward(cls, ctx: Context, d_out: float) -> Tuple[float, float]:
        """f'_x(x, y) = y + 1 ; f'_y(x, y) = x"""
        x, y = ctx.saved_values
        return operators.mul(d_out, operators.add(y, 1)), operators.mul(d_out, x)


class TensorFunction1(TensorFunction):
    @classmethod
    def _forward(cls, ctx: Context, x: Tensor, y: Tensor) -> Tensor:
        """f(x, y) = x + y + 10"""
        constant = tf.tensor([10])
        return x.func.add_zip(x, y.func.add_zip(y, constant))

    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return grad_out, grad_out


class TensorFunction2(TensorFunction):
    @classmethod
    def _forward(cls, ctx: Context, x: Tensor, y: Tensor) -> Tensor:
        """f(x, y) = x * y + x"""
        ctx.save_for_backward(x, y)
        return x.func.add_zip(x.func.mul_zip(x, y), x)

    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        """f'_x(x, y) = y + 1 ; f'_y(x, y) = x"""
        x, y = ctx.saved_tensors
        df_dx = grad_out.func.mul_zip(grad_out, y.func.add_zip(y, y._ensure_tensor(1)))
        df_dy = grad_out.func.mul_zip(grad_out, x)
        return df_dx, df_dy


def test_scalar_backprop1():
    """
    Example 1: F1(0.0, var1)
    """
    var1 = Scalar(0.0)
    var2 = ScalarFunction1.apply(0.0, var1)
    var2.backward(d_out=5.0)
    assert var1.derivative == 5.0


def test_scalar_backprop2():
    """
    Example 2: F1(0.0, F1(0.0, var1))
    """
    var1 = Scalar(0.0)
    var2 = ScalarFunction1.apply(0.0, var1)
    var3 = ScalarFunction1.apply(0.0, var2)
    var3.backward(d_out=5.0)
    assert var1.derivative == 5.0


def test_scalar_backprop3():
    """
    Example 3: F1(F1(0.0, var1), F1(0.0, var1))
    """
    var1 = Scalar(0.0)
    var2 = ScalarFunction1.apply(0.0, var1)
    var3 = ScalarFunction1.apply(0.0, var1)
    var4 = ScalarFunction1.apply(var2, var3)
    var4.backward(d_out=5.0)
    assert var1.derivative == 10.0


def test_scalar_backprop4():
    """
    Example 4: F1(F1(0.0, F1(0.0, var0), F1(0.0, F1(0.0, var0))
    """
    var0 = Scalar(0.0)
    var1 = ScalarFunction1.apply(0.0, var0)
    var2 = ScalarFunction1.apply(0.0, var1)
    var3 = ScalarFunction1.apply(0.0, var1)
    var4 = ScalarFunction1.apply(var2, var3)
    var4.backward(d_out=5.0)
    assert var0.derivative == 10.0


@pyt
def test_tensor_backprop1():
    # Create tensors
    shape = (5, 3)
    x = tf.ones(shape=shape)
    x.requires_grad = True
    y = tf.zeros(shape=shape)
    y.requires_grad = True

    # Forward
    out = TensorFunction1.apply(x, y)

    # Backward
    grad_out = Tensor.make(
        [5 for _ in range(out.size)],
        shape=out.shape,
        backend=SimpleBackend,
    )
    out.backward(grad_out=grad_out)

    # Check derivatives
    assert x.derivative is not None
    assert x.derivative.shape == x.shape
    assert np.all(
        np.array(x.derivative.data.storage) == np.array(grad_out.data.storage)
    )

    assert y.derivative is not None
    assert y.derivative.shape == y.shape
    assert np.all(
        np.array(y.derivative.data.storage) == np.array(grad_out.data.storage)
    )


def test_tensor_backprop2():
    # Create tensors
    x = tf.ones(shape=(3,))
    x.requires_grad = True
    y = tf.zeros(shape=(5, 1))
    y.requires_grad = True
    z = tf.ones(shape=(1, 3))
    z.requires_grad = True

    # Forward
    out1 = TensorFunction1.apply(x, y)
    out2 = TensorFunction1.apply(out1, z)

    # Backward
    grad_out = Tensor.make(
        [5 for _ in range(out2.size)],
        shape=out2.shape,
        backend=SimpleBackend,
    )
    out2.backward(grad_out=grad_out)

    # Check derivatives
    expected_grad = np.array([25 for _ in range(x.size)])
    assert x.derivative is not None
    assert x.derivative.shape == x.shape
    assert np.all(np.array(x.derivative.data.storage) == expected_grad)

    expected_grad = np.array([15 for _ in range(y.size)])
    assert y.derivative is not None
    assert y.derivative.shape == y.shape
    assert np.all(np.array(y.derivative.data.storage) == expected_grad)

    expected_grad = np.array([25 for _ in range(z.size)])
    assert z.derivative is not None
    assert z.derivative.shape == z.shape
    assert np.all(np.array(z.derivative.data.storage) == expected_grad)


def test_tensor_backprop3():
    # Create tensors
    x = tf.ones(shape=(3,)) * 0.5
    x.requires_grad = True
    y = tf.ones(shape=(5, 1)) * 0.75
    y.requires_grad = True

    # Forward
    out = TensorFunction2.apply(x, y)

    # Backward
    grad_out = Tensor.make(
        [0.5 for _ in range(out.size)],
        shape=out.shape,
        backend=SimpleBackend,
    )
    out.backward(grad_out=grad_out)

    # Check derivatives
    expected_grad = np.array([4.375 for _ in range(x.size)])
    assert x.derivative is not None
    assert x.derivative.shape == x.shape
    assert np.all(np.array(x.derivative.data.storage) == expected_grad)

    expected_grad = np.array([0.75 for _ in range(y.size)])
    assert y.derivative is not None
    assert y.derivative.shape == y.shape
    assert np.all(np.array(y.derivative.data.storage) == expected_grad)


def test_tensor_backprop4():
    # Create tensors
    x = tf.ones(shape=(3,)) * 0.5
    x.requires_grad = True
    y = tf.ones(shape=(5, 1)) * 0.75
    y.requires_grad = True
    z = tf.ones(shape=(1, 3)) * 0.5
    z.requires_grad = True

    # Forward
    out1 = TensorFunction2.apply(x, y)
    out2 = TensorFunction2.apply(out1, z)

    # Backward
    grad_out = Tensor.make(
        [0.5 for _ in range(out2.size)],
        shape=out2.shape,
        backend=SimpleBackend,
    )
    out2.backward(grad_out=grad_out)

    # Check derivatives
    expected_grad = np.array([6.5625 for _ in range(x.size)])
    assert x.derivative is not None
    assert x.derivative.shape == x.shape
    assert np.all(np.array(x.derivative.data.storage) == expected_grad)

    expected_grad = np.array([1.125 for _ in range(y.size)])
    assert y.derivative is not None
    assert y.derivative.shape == y.shape
    assert np.all(np.array(y.derivative.data.storage) == expected_grad)

    expected_grad = np.array([2.1875 for _ in range(z.size)])
    assert z.derivative is not None
    assert z.derivative.shape == z.shape
    assert np.all(np.array(z.derivative.data.storage) == expected_grad)


def test_tensor_backprop5():
    # Create tensors
    x = tf.ones(shape=(3,)) * 0.5
    x.requires_grad = True
    y = tf.ones(shape=(5, 1)) * 0.75
    y.requires_grad = True
    z = tf.ones(shape=(1, 3)) * 0.5
    z.requires_grad = True
    w = tf.ones(shape=(5, 3)) * 0.25
    w.requires_grad = True

    # Forward
    out1 = TensorFunction2.apply(x, y)
    out2 = TensorFunction2.apply(out1, z)
    out3 = TensorFunction2.apply(out2, w)

    # Backward
    grad_out = Tensor.make(
        [0.5 for _ in range(out3.size)],
        shape=out3.shape,
        backend=SimpleBackend,
    )
    out3.backward(grad_out=grad_out)

    # Check derivatives - calculated from torch.
    expected_grad = np.array([8.2031 for _ in range(x.size)])
    assert x.derivative is not None
    assert x.derivative.shape == x.shape
    assert np.allclose(np.array(x.derivative.data.storage), expected_grad)

    expected_grad = np.array([1.40625 for _ in range(y.size)])
    assert y.derivative is not None
    assert y.derivative.shape == y.shape
    assert np.allclose(np.array(y.derivative.data.storage), expected_grad)

    expected_grad = np.array([2.7344 for _ in range(z.size)])
    assert z.derivative is not None
    assert z.derivative.shape == z.shape
    assert np.allclose(np.array(z.derivative.data.storage), expected_grad)

    expected_grad = np.array([0.65625 for _ in range(w.size)])
    assert w.derivative is not None
    assert w.derivative.shape == w.shape
    assert np.allclose(np.array(w.derivative.data.storage), expected_grad)
