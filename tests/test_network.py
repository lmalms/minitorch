import os
import random

import numpy as np
import pytest
from hypothesis import given

from minitorch.module import Network
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
def test_network_forward_floats(input_dim: int, hidden_dim: int, output_dim: int):
    network = Network(input_dim, hidden_dim, output_dim)

    # Make up some data
    n_samples = 100
    X = [[i * random.random() for i in range(input_dim)] for _ in range(n_samples)]
    y_hat = np.array([[scalar.data for scalar in row] for row in network.forward(X)])

    # Compare to using numpy and operators
    # Input to hidden layer
    ih_weights = np.array(
        [[param.value.data for param in row] for row in network._input_layer._weights]
    )
    ih_bias = np.array([param.value.data for param in network._input_layer._bias])
    ih_state = np.dot(np.array(X), ih_weights) + ih_bias
    ih_state = [[relu(x) for x in row] for row in ih_state]

    # Hidden to hidden layer
    hh_weights = np.array(
        [[param.value.data for param in row] for row in network._hidden_layer._weights]
    )
    hh_bias = np.array([param.value.data for param in network._hidden_layer._bias])
    hh_state = np.dot(np.array(ih_state), hh_weights) + hh_bias
    hh_state = [[relu(x) for x in row] for row in hh_state]

    # Hidden to output layer
    ho_weights = np.array(
        [[param.value.data for param in row] for row in network._output_layer._weights]
    )
    ho_bias = np.array([param.value.data for param in network._output_layer._bias])
    ho_state = np.dot(np.array(hh_state), ho_weights) + ho_bias
    hh_state = [[relu(x) for x in row] for row in ho_state]

    assert np.all(np.isclose(y_hat, hh_state))
