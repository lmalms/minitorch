import random
from typing import List, Union

import numpy as np
import pytest
from hypothesis import given

import minitorch.autodiff.tensor_functions as tf
from minitorch.autodiff import Scalar, Tensor
from minitorch.module import (
    LinearScalarLayer,
    LinearTensorLayer,
    ScalarNetwork,
    TensorNetwork,
)
from minitorch.operators import relu

from .strategies import medium_ints

SKIP_NETWORK_FORWARD_TESTS = True
SKIP_REASON = "Tests are slow."


@given(medium_ints, medium_ints, medium_ints)
def test_scalar_network_init(input_dim: int, hidden_dim: int, output_dim: int):
    network = ScalarNetwork(input_dim, hidden_dim, output_dim)

    # Check dimensions of weight and bias matrices
    # Input layer
    assert len(network._input_layer._weights) == input_dim
    assert len(network._input_layer._weights[0]) == hidden_dim
    assert len(network._input_layer._bias) == hidden_dim

    # Hidden layer
    assert len(network._hidden_layer._weights) == hidden_dim
    assert len(network._hidden_layer._weights[0]) == hidden_dim
    assert len(network._hidden_layer._bias) == hidden_dim

    # Output layer
    assert len(network._output_layer._weights) == hidden_dim
    assert len(network._output_layer._weights[0]) == output_dim
    assert len(network._output_layer._bias) == output_dim


@given(medium_ints, medium_ints, medium_ints)
@pytest.mark.skipif(SKIP_NETWORK_FORWARD_TESTS, reason=SKIP_REASON)
def test_scalar_network_forward(input_dim: int, hidden_dim: int, output_dim: int):

    # Utils functions for running tests
    def extract_weights_and_biases(layer: LinearScalarLayer):
        weights = np.array(
            [[param.value.data for param in row] for row in layer._weights]
        )
        biases = np.array([param.value.data for param in layer._bias])
        return weights, biases

    def minitorch_forward(input_: List[List[Union[float, Scalar]]]) -> np.ndarray:
        y_hat = network.forward(input_)
        return np.array([[scalar.data for scalar in row] for row in y_hat])

    def np_forward(input_: List[List[float]]) -> np.ndarray:
        # Input to hidden
        ih_state = np.dot(np.array(input_), ih_weights) + ih_bias
        ih_state = np.array([[relu(x) for x in row] for row in ih_state])

        # Hidden to hidden
        hh_state = np.dot(np.array(ih_state), hh_weights) + hh_bias
        hh_state = np.array([[relu(x) for x in row] for row in hh_state])

        # Hidden to out
        ho_state = np.dot(np.array(hh_state), ho_weights) + ho_bias
        return ho_state

    # Initialise network
    network = ScalarNetwork(input_dim, hidden_dim, output_dim)

    # Extract weights and biases
    ih_weights, ih_bias = extract_weights_and_biases(network._input_layer)
    hh_weights, hh_bias = extract_weights_and_biases(network._hidden_layer)
    ho_weights, ho_bias = extract_weights_and_biases(network._output_layer)

    n_samples = 10
    X = [[i * random.random() for i in range(input_dim)] for _ in range(n_samples)]
    assert np.all(np.isclose(minitorch_forward(X), np_forward(X)))

    X_scalars = [[Scalar(value) for value in row] for row in X]
    assert np.allclose(minitorch_forward(X_scalars), np_forward(X))


@given(medium_ints, medium_ints, medium_ints)
def test_tensor_network_init(input_dim: int, hidden_dim: int, output_dim: int):
    network = TensorNetwork(input_dim, hidden_dim, output_dim)

    # Check dimensions of weight and bias matrices
    # Input layer
    in_weights = network._input_layer._weights
    in_bias = network._input_layer._bias
    assert in_weights.value.shape == (input_dim, hidden_dim)
    assert in_bias.value.shape == (hidden_dim,)

    # Hidden layer
    hidden_weights = network._hidden_layer._weights
    hidden_bias = network._hidden_layer._bias
    assert hidden_weights.value.shape == (hidden_dim, hidden_dim)
    assert hidden_bias.value.shape == (hidden_dim,)

    # Output layer
    out_weights = network._output_layer._weights
    out_bias = network._output_layer._bias
    assert out_weights.value.shape == (hidden_dim, output_dim)
    assert out_bias.value.shape == (output_dim,)


@given(medium_ints, medium_ints, medium_ints)
def test_tensor_network_forward(input_dim: int, hidden_dim: int, output_dim: int):
    def extract_weights_and_biases(layer: LinearTensorLayer):
        weights = layer._weights.value
        weights = np.array(weights.data.storage).reshape(weights.shape)

        bias = layer._bias.value
        bias = np.array(bias.data.storage).reshape(bias.shape)

        return weights, bias

    def minitorch_forward(inputs: Tensor) -> np.ndarray:
        out_ = network.forward(inputs)
        return np.array(out_.data.storage).reshape(out_.shape)

    def numpy_forward(inputs: Tensor) -> np.ndarray:
        def apply_relu(inputs: np.ndarray) -> np.ndarray:
            out = [relu(i) for i in inputs.flatten()]
            return np.array(out).reshape(inputs.shape)

        # To numpy
        inputs = np.array(inputs.data.storage).reshape(inputs.shape)
        ih_state = apply_relu(np.dot(inputs, ih_weights) + ih_bias)
        hh_state = apply_relu(np.dot(ih_state, hh_weights) + hh_bias)
        ho_state = np.dot(hh_state, ho_weights) + ho_bias
        return ho_state

    # Initialise network
    network = TensorNetwork(input_dim, hidden_dim, output_dim)

    # Extract weights and biases
    ih_weights, ih_bias = extract_weights_and_biases(network._input_layer)
    hh_weights, hh_bias = extract_weights_and_biases(network._hidden_layer)
    ho_weights, ho_bias = extract_weights_and_biases(network._output_layer)

    # Generate some data and run forwards
    n_samples = 10
    inputs = tf.rand((n_samples, input_dim))
    assert np.allclose(minitorch_forward(inputs), numpy_forward(inputs))
