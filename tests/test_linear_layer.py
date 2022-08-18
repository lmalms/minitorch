import random

import numpy as np
import pytest
from hypothesis import given

from minitorch.autodiff import Scalar
from minitorch.module import Linear

from .strategies import medium_ints

SKIP_LINEAR_FORWARD_TESTS = True
SKIP_REASON = "Tests are slow."


@given(medium_ints, medium_ints)
def test_linear_init(input_dim: int, output_dim: int):
    linear = Linear(input_dim, output_dim)

    # Check the size and dim of weight matrix
    assert len(linear._weights) == input_dim
    assert all(len(weights) == output_dim for weights in linear._weights)

    # Check size and dim of bias matrix
    assert len(linear._bias) == output_dim


@given(medium_ints, medium_ints)
@pytest.mark.skipif(SKIP_LINEAR_FORWARD_TESTS, reason=SKIP_REASON)
def test_linear_forward_floats(input_dim: int, output_dim: int):
    # Initialise a linear layer
    linear = Linear(input_dim, output_dim)

    # Make up some data and propagate forward
    n_samples = 100
    X = [[i * random.random() for i in range(input_dim)] for _ in range(n_samples)]
    y_hat = linear.forward(X)
    y_hat = np.array([[scalar.data for scalar in row] for row in y_hat])

    # Compare to np implementation
    weights = np.array([[param.value.data for param in row] for row in linear._weights])
    bias = np.array([param.value.data for param in linear._bias])
    y_hat_np = np.dot(np.array(X), weights) + bias

    assert np.allclose(y_hat, y_hat_np)


@given(medium_ints, medium_ints)
@pytest.mark.skipif(SKIP_LINEAR_FORWARD_TESTS, reason=SKIP_REASON)
def test_linear_forward_scalars(input_dim: int, output_dim: int):
    # Initialise a linear layer
    linear = Linear(input_dim, output_dim)

    # Make up some data and propagate forward
    n_samples = 100
    X = [
        [
            Scalar(
                value=i * random.random(),
            )
            for i in range(input_dim)
        ]
        for _ in range(n_samples)
    ]
    y_hat = linear.forward(X)
    y_hat = np.array([[scalar.data for scalar in row] for row in y_hat])

    # Compare to np implementation
    weights = np.array([[param.value.data for param in row] for row in linear._weights])
    bias = np.array([param.value.data for param in linear._bias])
    X_np = np.array([[input_.data for input_ in row] for row in X])
    y_hat_np = np.dot(np.array(X_np), weights) + bias

    assert np.allclose(y_hat, y_hat_np)
