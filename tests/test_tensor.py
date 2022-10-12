from typing import Callable, Iterable, List, Tuple

import pytest
from hypothesis import given
from hypothesis.strategies import DataObject, data, lists, permutations

from minitorch.autodiff import Tensor, tensor
from minitorch.testing import MathTestVariable

from .strategies import small_floats
from .tensor_strategies import shaped_tensors, tensors

# Check that MathTestVariable is good to go for tensors as well!


@given(lists(small_floats, min_size=1))
def test_create_and_index(data: List[float]) -> None:
    """
    Test 1D tensor creation and indexing
    """
    t = tensor(data)
    for idx in range(len(data)):
        assert data[idx] == t[idx]
