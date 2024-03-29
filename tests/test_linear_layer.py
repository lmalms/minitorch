import random
from functools import partial, update_wrapper
from typing import List, Union

import numpy as np
import pytest
from hypothesis import given, settings

import minitorch.autodiff.tensor_functions as tf
from minitorch.autodiff import FastOps, Scalar, SimpleOps, Tensor, TensorBackend
from minitorch.module import LinearScalarLayer, LinearTensorLayer

from .strategies import medium_ints

# Define backends
BACKENDS = {"simple": TensorBackend(SimpleOps), "fast": TensorBackend(FastOps)}


@given(medium_ints, medium_ints)
def test_linear_scalar_init(input_dim: int, output_dim: int):
    linear = LinearScalarLayer(input_dim, output_dim)

    # Check the size and dim of weight matrix
    assert len(linear._weights) == input_dim
    assert all(len(weights) == output_dim for weights in linear._weights)

    # Check size and dim of bias matrix
    assert len(linear._bias) == output_dim


@given(medium_ints, medium_ints)
@settings(max_examples=100)
def test_linear_scalar_forward(input_dim: int, output_dim: int):

    # Initialise a linear layer
    linear = LinearScalarLayer(input_dim, output_dim)
    weights = np.array([[param.value.data for param in row] for row in linear._weights])
    bias = np.array([param.value.data for param in linear._bias])

    def minitorch_forward(X: List[List[Union[float, Scalar]]]) -> np.ndarray:
        y_hat = linear.forward(X)
        return np.array([[scalar.data for scalar in row] for row in y_hat])

    def np_forward(X: List[List[float]]) -> np.ndarray:
        return np.dot(np.array(X), weights) + bias

    # Make up some data and compare to np implementation
    n_samples = 10
    X = [[i * random.random() for i in range(input_dim)] for _ in range(n_samples)]
    assert np.allclose(minitorch_forward(X), np_forward(X))

    X_scalars = [[Scalar(value) for value in row] for row in X]
    assert np.allclose(minitorch_forward(X_scalars), np_forward(X))


@given(medium_ints, medium_ints)
@settings(max_examples=100)
@pytest.mark.parametrize("backend", (pytest.param("simple"), pytest.param("fast")))
def test_linear_tensor_init(backend: str, input_dim: int, output_dim: int):
    linear = LinearTensorLayer(input_dim, output_dim, BACKENDS[backend])

    # Check shape of weights and biases
    assert linear._weights.value.shape == (input_dim, output_dim)
    assert linear._bias.value.shape == (output_dim,)


@pytest.mark.parametrize("backend", (pytest.param("simple"), pytest.param("fast")))
def test_linear_tensor_forward(backend: str, input_dim: int = 2, output_dim: int = 1):
    # Initialise a new linear layer
    linear = LinearTensorLayer(input_dim, output_dim, backend=BACKENDS[backend])
    weights, bias = linear._weights.value, linear._bias.value
    weights_np = np.array(weights.data.storage).reshape(weights.shape)
    bias_np = np.array(bias.data.storage).reshape(bias.shape)

    # Generate some input data
    n_samples = 10
    inputs = tf.rand((n_samples, input_dim), backend=BACKENDS[backend])
    inputs_np = np.array(inputs.data.storage).reshape((n_samples, input_dim))

    # Forward
    tensor_out = linear.zip_reduce_forward(inputs)
    np_out = np.dot(inputs_np, weights_np) + bias_np

    # Check zip reduce forward
    assert np.all(np.isclose(tensor_out.data.storage, np_out.flatten()))

    if backend == "fast":
        # Check matmul forward
        tensor_out = linear.forward(inputs)
        assert np.all(np.isclose(tensor_out.data.storage, np_out.flatten()))


@pytest.mark.parametrize("backend", (pytest.param("simple"), pytest.param("fast")))
def test_linear_tensor_backward(backend: str, input_dim: int = 2, output_dim: int = 1):
    def zip_reduce_forward(inputs: Tensor, weights: Tensor, bias: Tensor) -> Tensor:
        """
        Separate out zip reduce forward for grad_check
        """
        # Add dimensions such that we can broadcast
        inputs = inputs.view(*inputs.shape, 1)
        weights = weights.view(1, *weights.shape)

        # Collapse dimension
        out = (inputs * weights).sum(dim=1)
        out = out.view(inputs.shape[0], bias.shape[0])
        return out + bias

    def mat_mul_forward(inputs: Tensor, weights: Tensor, bias: Tensor) -> Tensor:
        """
        Separate out mat mul forward for grad check
        """
        return inputs @ weights + bias

    # Initialise a new linear layer
    weights_parameter = LinearTensorLayer._initialise_parameter(
        input_dim,
        output_dim,
        backend=BACKENDS[backend],
    )
    bias_parameter = LinearTensorLayer._initialise_parameter(
        output_dim,
        backend=BACKENDS[backend],
    )
    weights = weights_parameter.value
    bias = bias_parameter.value

    # Generate some input data
    n_samples = 10
    inputs = tf.rand((n_samples, input_dim), backend=BACKENDS[backend])

    zip_red_f = update_wrapper(partial(zip_reduce_forward, inputs), zip_reduce_forward)
    mat_mul_f = update_wrapper(partial(mat_mul_forward, inputs), mat_mul_forward)

    # Check zip reduce forward
    tf.grad_check(zip_red_f, weights, bias)

    # Check mat mul forward
    if backend == "fast":
        tf.grad_check(mat_mul_f, weights, bias)


@pytest.mark.parametrize("backend", (pytest.param("simple"), pytest.param("fast")))
def test_linear_tensor_backward_with_loss(
    backend: str, input_dim: int = 2, output_dim: int = 1
):
    def zip_reduce_forward(
        inputs: Tensor,
        targets: Tensor,
        weights: Tensor,
        bias: Tensor,
    ) -> Tensor:
        """
        Separate out zip reduce forward for grad check.
        """
        # Add dimensions such that we can broadcast
        inputs = inputs.view(*inputs.shape, 1)
        weights_ = weights.view(1, *weights.shape)

        # Collapse dimension
        out = (inputs * weights_).sum(dim=1)
        logits = out.view(inputs.shape[0], bias.size) + bias
        predictions = logits.sigmoid().view(targets.size)

        # Compute loss
        log_likelihoods_p = (targets == 1) * predictions.log()
        log_likelihoods_f = (targets == 0) * (-predictions + 1).log()
        loss = (log_likelihoods_p + log_likelihoods_f) / float(predictions.shape[0])

        return loss

    def mat_mul_forward(
        inputs: Tensor,
        targets: Tensor,
        weights: Tensor,
        bias: Tensor,
    ) -> Tensor:
        """
        Separate out zip reduce forward for grad check.
        """
        logits = inputs @ weights + bias
        predictions = logits.sigmoid().view(targets.size)

        # Compute loss
        log_likelihoods_p = (targets == 1) * predictions.log()
        log_likelihoods_f = (targets == 0) * (-predictions + 1).log()
        loss = (log_likelihoods_p + log_likelihoods_f) / float(predictions.shape[0])

        return loss

    # Initialise a new linear layer
    weights_parameter = LinearTensorLayer._initialise_parameter(
        input_dim,
        output_dim,
        backend=BACKENDS[backend],
    )
    bias_parameter = LinearTensorLayer._initialise_parameter(
        output_dim,
        backend=BACKENDS[backend],
    )
    weights = weights_parameter.value
    bias = bias_parameter.value

    # Generate some input data
    n_samples = 10
    inputs = tf.rand((n_samples, input_dim), backend=BACKENDS[backend])
    targets = tf.tensor([1, 1, 1, 0, 0, 0, 1, 0, 1, 0], backend=BACKENDS[backend])

    # Check zip reduce forward
    zip_red_f = partial(zip_reduce_forward, inputs, targets)
    zip_red_f = update_wrapper(zip_red_f, zip_reduce_forward)
    tf.grad_check(zip_red_f, weights, bias)

    # Check mat mul forward
    if backend == "fast":
        mat_mul_f = partial(mat_mul_forward, inputs, targets)
        mat_mul_f = update_wrapper(mat_mul_f, mat_mul_forward)
        tf.grad_check(mat_mul_f, weights, bias)
