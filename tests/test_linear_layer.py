import os
import random
from typing import List, Union

import numpy as np
import pytest
from hypothesis import given

from minitorch.autodiff import Scalar
from minitorch.autodiff import tensor_functions as tf
from minitorch.module import LinearScalar, LinearTensor

from .strategies import medium_ints

SKIP_LINEAR_FORWARD_TESTS = True
SKIP_REASON = "Tests are slow."


@given(medium_ints, medium_ints)
def test_linear_scalar_init(input_dim: int, output_dim: int):
    linear = LinearScalar(input_dim, output_dim)

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
def test_linear_scalar_forward(input_dim: int, output_dim: int):

    # Initialise a linear layer
    linear = LinearScalar(input_dim, output_dim)
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


@given(medium_ints, medium_ints)
def test_linear_tensor_init(input_dim: int, output_dim: int):
    linear = LinearTensor(input_dim, output_dim)

    # Check shape of weights and biases
    assert linear._weights.value.shape == (input_dim, output_dim)
    assert linear._bias.value.shape == (output_dim,)


@given(medium_ints, medium_ints)
def test_linear_scalar_forward(input_dim: int, output_dim: int):
    # Initialise a new linear layer
    linear = LinearTensor(input_dim, output_dim)
    weights, bias = linear._weights.value, linear._bias.value
    weights_np = np.array(weights.data.storage).reshape(weights.shape)
    bias_np = np.array(bias.data.storage).reshape(bias.shape)

    # Generate some input data
    n_samples = 10
    inputs = tf.rand((n_samples, input_dim))
    inputs_np = np.array(inputs.data.storage).reshape((n_samples, input_dim))

    # Forward
    tensor_out = linear.forward(inputs=inputs)
    np_out = np.dot(inputs_np, weights_np) + bias_np

    # Check
    assert np.all(np.isclose(tensor_out.data.storage, np_out.flatten()))
