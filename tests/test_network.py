import numpy as np
from hypothesis import given

from minitorch.module import Network

from .strategies import medium_ints

SKIP_NETWORK_TESTS = False
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
def test_network_forward(input_dim: int, hidden_dim: int, output_dim: int):
    pass
