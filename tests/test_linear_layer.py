import numpy as np
from hypothesis import given

from minitorch.module import Linear

from .strategies import medium_ints


@given(medium_ints, medium_ints)
def test_init(input_dim: int, output_dim: int):
    linear = Linear(input_dim, output_dim)

    # Check the size and dim of weight matrix
    assert len(linear._weights) == input_dim
    assert all(len(weights) == output_dim for weights in linear._weights)

    # Check size and dim of bias matrix
    assert len(linear._bias) == output_dim


def test_forward():
    pass