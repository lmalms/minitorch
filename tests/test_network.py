import os
import random
from typing import List, Union

import numpy as np
import pytest
from hypothesis import given

from minitorch.autodiff import Scalar
from minitorch.module import Linear, Network
from minitorch.operators import relu

from .strategies import medium_ints

SKIP_NETWORK_FORWARD_TESTS = False
SKIP_REASON = "Tests are slow."


@given(medium_ints, medium_ints, medium_ints)
def test_network_init(input_dim: int, hidden_dim: int, output_dim: int):
    network = Network(input_dim, hidden_dim, output_dim)

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
@pytest.mark.skipif(
    os.environ.get("SKIP_NETWORK_FORWARD_TESTS", SKIP_NETWORK_FORWARD_TESTS),
    reason=SKIP_REASON,
)
def test_network_forward(input_dim: int, hidden_dim: int, output_dim: int):

    # Utils functions for running tests
    def extract_weights_and_biases(layer: Linear):
        weights = np.array(
            [[param.value.data for param in row] for row in layer._weights]
        )
        biases = np.array([param.value.data for param in layer._bias])
        return weights, biases

    def minitorch_forward(input_: List[List[Union[float, Scalar]]]) -> np.ndarray:
        y_hat = network.forward(input_)
        return np.array([[scalar.data for scalar in row] for row in y_hat])

    def np_forward_single(
        weights: np.ndarray, biases: np.ndarray, input_: List[List[float]]
    ) -> np.ndarray:
        out_ = np.dot(np.array(input_), weights) + biases
        out_ = [[relu(x) for x in row] for row in out_]
        return np.array(out_)

    def np_forward(input_: List[List[float]]) -> np.ndarray:
        ih_state = np_forward_single(ih_weights, ih_bias, input_)
        hh_state = np_forward_single(hh_weights, hh_bias, ih_state)
        ho_state = np_forward_single(ho_weights, ho_bias, hh_state)
        return ho_state

    # Initialise network
    network = Network(input_dim, hidden_dim, output_dim)

    # Extract weights and biases
    ih_weights, ih_bias = extract_weights_and_biases(network._input_layer)
    hh_weights, hh_bias = extract_weights_and_biases(network._hidden_layer)
    ho_weights, ho_bias = extract_weights_and_biases(network._output_layer)

    n_samples = 100
    X = [[i * random.random() for i in range(input_dim)] for _ in range(n_samples)]
    assert np.all(np.isclose(minitorch_forward(X), np_forward(X)))

    X_scalars = [[Scalar(value) for value in row] for row in X]
    assert np.allclose(minitorch_forward(X_scalars), np_forward(X))
