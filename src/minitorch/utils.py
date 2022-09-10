
from typing import List
from minitorch.autodiff import Scalar


def check_for_equal_dim(y_true: List[Scalar], y_hat: List[Scalar]):
    assert len(y_true) == len(y_hat), "y_true and y_hat need to have the same length."
