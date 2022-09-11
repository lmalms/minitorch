from typing import List

from minitorch.autodiff import Scalar


def check_for_equal_dim(y_true: List[Scalar], y_hat: List[Scalar]):
    assert len(y_true) == len(y_hat), "y_true and y_hat need to have the same length."


def check_for_binary_values(y: List[Scalar]):
    def _is_binary(y: List[Scalar]) -> bool:
        return all((s.data == 0) or (s.data == 1) for s in y)

    assert _is_binary(y), "y can only contain scalars of value 1. or 0."
