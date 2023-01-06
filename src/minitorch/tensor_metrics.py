from typing import Tuple

import numpy as np
from typing_extensions import TypeAlias

from minitorch.autodiff import Tensor, tensor

RocCurve: TypeAlias = Tuple[Tensor, Tensor, Tensor]


def _check_dims(y_true: Tensor, y_hat: Tensor):
    assert y_true.shape == y_hat.shape, "tensors need to have the same shape."
    assert y_true.dims == y_hat.dims == 1, "tensor should be 1D (n_samples, )."


def _check_for_binary_values(y: Tensor):
    def is_binary(y: Tensor) -> bool:
        all_binary = ((y == 0.0) + (y == 1.0)).all().item()
        return bool(all_binary)

    assert is_binary(y), "y can only contain values of 1. or 0."


def _check_arrays(y_true: Tensor, y_hat: Tensor) -> None:
    _check_dims(y_true, y_hat)
    _check_for_binary_values(y_true)
    _check_for_binary_values(y_hat)


def accuracy(y_true: Tensor, y_hat: Tensor) -> Tensor:
    _check_arrays(y_true, y_hat)
    n_correct = (y_true == y_hat).sum()
    return n_correct / y_true.size


def true_positive_rate(y_true: Tensor, y_hat: Tensor) -> Tensor:
    _check_arrays(y_true, y_hat)
    tp = ((y_hat == 1.0) * (y_true == 1.0)).sum()
    fn = ((y_hat == 0.0) * (y_true == 1.0)).sum()
    return tp / (tp + fn)


def false_positive_rate(y_true: Tensor, y_hat: Tensor) -> Tensor:
    _check_arrays(y_true, y_hat)
    fp = ((y_hat == 1.0) * (y_true == 0.0)).sum()
    tn = ((y_hat == 0.0) * (y_true == 0.0)).sum()
    return fp / (fp + tn)


def precision(y_true: Tensor, y_hat: Tensor) -> Tensor:
    _check_arrays(y_true, y_hat)
    tp = ((y_hat == 1.0) * (y_true == 1.0)).sum()
    fp = ((y_hat == 1.0) * (y_true == 0.0)).sum()
    return tp / (tp + fp)


def recall(y_true: Tensor, y_hat: Tensor) -> Tensor:
    return true_positive_rate(y_true, y_hat)


def sensitivity(y_true: Tensor, y_hat: Tensor) -> Tensor:
    return true_positive_rate(y_true, y_hat)


def specificity(y_true: Tensor, y_hat: Tensor) -> Tensor:
    return 1 - false_positive_rate(y_true, y_hat)


def roc_curve(y_true: Tensor, y_hat: Tensor, bucket_size: float = 1e-03) -> RocCurve:
    """
    Computes ROC curve, that is a plot of true positive and false positive rates as a function of classification
    threshold.

    Args:
         y_true - Tensor
            The true probabilities for the positive class.
        y_hat - Tensor
            The predicted probabilities for the positive class.
        bucket_size - float, default = 1e-03
            The threshold interval at which to compute the true positive and false positive rates.
            Note: Has to be between 1e-03 and 1.
    """
    _check_dims(y_true, y_hat)

    def bucket_thresholds(y_hat: np.ndarray, bucket_size: float = 1e-03):
        bucket_size = min(max(bucket_size, 0.001), 1.0)
        thresholds = [0.0]
        for y_h in y_hat:
            # If y_hat is within bucket_size of max threshold, move on
            # Otherwise append y_hat to make new max threshold
            if max(thresholds) < (y_h - bucket_size):
                thresholds.append(y_h)

        return thresholds

    # Sort because will be iterting through thresholds in increasing order
    y_true, y_hat = y_true.data.storage, y_hat.data.storage
    sorted_pairs = sorted(zip(y_true, y_hat), key=lambda pair: pair[1])
    y_true, y_hat = zip(*[(y_t, y_h) for (y_t, y_h) in sorted_pairs])
    thresholds = bucket_thresholds(y_hat, bucket_size)

    tpr, fpr = [], []
    for threshold in thresholds:
        y_hat_at_threshold = [1.0 if (proba >= threshold) else 0.0 for proba in y_hat]
        y_hat_at_threshold = tensor(y_hat_at_threshold)
        tpr_at_threshold = true_positive_rate(tensor(y_true), y_hat_at_threshold)
        fpr_at_threshold = false_positive_rate(tensor(y_true), y_hat_at_threshold)
        tpr.append(tpr_at_threshold.item())
        fpr.append(fpr_at_threshold.item())

    return tensor(tpr), tensor(fpr), tensor(thresholds)
