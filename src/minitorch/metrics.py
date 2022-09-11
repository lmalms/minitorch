from typing import List

from minitorch.autodiff import Scalar
from minitorch.functional import summation
from minitorch.types import RocCurveResult
from minitorch.utils import check_for_binary_values, check_for_equal_dim


def _check_arrays(y_true: List[Scalar], y_hat: List[Scalar]) -> None:
    check_for_equal_dim(y_true, y_hat)
    check_for_binary_values(y_true)
    check_for_binary_values(y_hat)


def accuracy(y_true: List[Scalar], y_hat: List[Scalar]) -> Scalar:
    _check_arrays(y_true, y_hat)
    n_correct = summation([t == p for (t, p) in zip(y_true, y_hat)])
    return n_correct / len(y_true)


def true_positive_rate(y_true: List[Scalar], y_hat: List[Scalar]) -> Scalar:
    _check_arrays(y_true, y_hat)
    tp = summation([(t == p) and (t == 1) for (t, p) in zip(y_true, y_hat)])
    fn = summation([(t != p) and (t == 1) for (t, p) in zip(y_true, y_hat)])
    return tp / (tp + fn)


def false_positive_rate(y_true: List[Scalar], y_hat: List[Scalar]) -> Scalar:
    _check_arrays(y_true, y_hat)
    fp = summation([(t != p) and (t == 0) for (t, p) in zip(y_true, y_hat)])
    tn = summation([(t == p) and (t == 0) for (t, p) in zip(y_true, y_hat)])
    return fp / (fp + tn)


def precision(y_true: List[Scalar], y_hat: List[Scalar]) -> Scalar:
    _check_arrays(y_true, y_hat)
    tp = summation([(t == p) and (t == 1) for (t, p) in zip(y_true, y_hat)])
    fp = summation([(t != p) and (t == 0) for (t, p) in zip(y_true, y_hat)])
    return tp / (tp + fp)


def recall(y_true: List[Scalar], y_hat: List[Scalar]) -> Scalar:
    return true_positive_rate(y_true, y_hat)


def sensitivity(y_true: List[Scalar], y_hat: List[Scalar]) -> Scalar:
    return true_positive_rate(y_true, y_hat)


def specificity(y_true: List[Scalar], y_hat: List[Scalar]) -> Scalar:
    return 1 - false_positive_rate(y_true, y_hat)


def roc_curve(
    y_true: List[Scalar], y_hat: List[Scalar], bucket_size: float = 1e-03
) -> RocCurveResult:
    """
    Computes ROC curve, that is a plot of true positive and false positive rates as a function of classification
    threshold.

    Args:
         y_true - List[Scalar]
            The true probabilities for the positive class.
        y_hat - List[Scalar]
            The predicted probabilities for the positive class.
        bucket_size - float, default = 1e-03
            The threshold interval at which to compute the true positive and false positive rates.
            Note: Has to be between 1e-03 and 1.
    """
    check_for_equal_dim(y_true, y_hat)

    def bucket_thresholds(y_hat: List[Scalar], bucket_size: float = 1e-03):
        bucket_size = min(max(bucket_size, 0.001), 1.0)
        thresholds = [0.0]
        for y_h in y_hat:
            if max(thresholds) < (y_h - bucket_size):
                thresholds.append(y_h)

        return thresholds

    sorted_pairs = sorted(zip(y_true, y_hat), key=lambda pair: pair[1])
    y_true, y_hat = zip(*[(y_t, y_h) for (y_t, y_h) in sorted_pairs])
    thresholds = bucket_thresholds(y_hat, bucket_size)

    tpr, fpr = [], []
    for threshold in thresholds:
        y_hat_at_threshold = [
            Scalar(1.0) if (proba >= threshold) else Scalar(0.0) for proba in y_hat
        ]
        tpr_at_threshold = true_positive_rate(y_true, y_hat_at_threshold)
        fpr_at_threshold = false_positive_rate(y_true, y_hat_at_threshold)
        tpr.append(tpr_at_threshold)
        fpr.append(fpr_at_threshold)

    return tpr, fpr, thresholds


def precision_recall_curve():
    pass


def auc():
    pass
