import os
import random
from typing import List, Union

import numpy as np
import pytest
from hypothesis import given

from minitorch.autodiff import Scalar
from minitorch.module import Linear

from .strategies import medium_ints

SKIP_LINEAR_FORWARD_TESTS = False
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
@pytest.mark.skipif(
    os.environ.get("SKIP_LINEAR_FORWARD_TESTS", SKIP_LINEAR_FORWARD_TESTS),
    reason=SKIP_REASON,
)
def test_linear_forward(input_dim: int, output_dim: int):

    # Initialise a linear layer
    linear = Linear(input_dim, output_dim)
    weights = np.array([[param.value.data for param in row] for row in linear._weights])
    bias = np.array([param.value.data for param in linear._bias])

    def minitorch_forward(X: List[List[Union[float, Scalar]]]) -> np.ndarray:
        y_hat = linear.forward(X)
        return np.array([[scalar.data for scalar in row] for row in y_hat])

    def np_forward(X: List[List[float]]) -> np.ndarray:
        return np.dot(np.array(X), weights) + bias

    # Make up some data and compare to np implementation
    n_samples = 100
    X = [[i * random.random() for i in range(input_dim)] for _ in range(n_samples)]
    assert np.allclose(minitorch_forward(X), np_forward(X))

    X_scalars = [[Scalar(value) for value in row] for row in X]
    assert np.allclose(minitorch_forward(X_scalars), np_forward(X))
