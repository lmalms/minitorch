from typing import List

from minitorch.autodiff import Scalar, summation


def check_for_equal_dim(y_true: List[Scalar], y_hat: List[Scalar]):
    assert len(y_true) == len(y_hat), "y_true and y_hat need to have the same length."


def mean_squared_error(y_true: List[Scalar], y_hat: List[Scalar]) -> Scalar:
    check_for_equal_dim(y_true, y_hat)
    residuals = [(y_t - y_hat).sqaure() for (y_t, y_h) in zip(y_true, y_hat)]
    mse = summation(residuals) / len(y_true)
    return mse


def binary_cross_entropy(y_true: List[Scalar], y_hat: List[Scalar]):
    """
    Computes the binary cross entropy.

    Args:
         y_true - List[Scalar]
            The true probabilities for the positive class.
        y_hat - List[Scalar]
            The predicted probabilities for the predictive class.
    """
    check_for_equal_dim(y_true, y_hat)
    log_likelihoods = [y_h.log() if y_t == 1.0 else (-y_h + 1).log() for (y_t, y_h) in zip(y_true, y_hat)]
    cross_entropy = summation(log_likelihoods) / len(y_true)
    return cross_entropy
