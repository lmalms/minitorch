from typing import List, Tuple

from minitorch.autodiff import Scalar
from minitorch.functional import summation
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
    y_true: List[Scalar], y_hat: List[Scalar]
) -> Tuple[List[Scalar], List[Scalar], List[Scalar]]:
    # y_hat here should be the predicted probabilities of a particular
    # sample belonging to the positive class
    check_for_equal_dim(y_true, y_hat)

    # Infer the thresholds at which to compute tpr and fpr
    delta = 1e-06
    # TODO: Can I bucket the thresholds somehow?
    # 1. sort in ascending order
    y_true, y_hat = zip(
        *[
            (y_t, y_h)
            for (y_t, y_h) in sorted(zip(y_true, y_hat), key=lambda pair: pair[1])
        ]
    )
    thresholds = [y - delta for y in y_hat]
    tpr, fpr = [], []
    for threshold in thresholds:
        y_hat_at_threshold = [1.0 if (proba >= threshold) else 0.0 for proba in y_hat]
        tpr_at_threshold = true_positive_rate(y_true, y_hat_at_threshold)
        fpr_at_threshold = false_positive_rate(y_true, y_hat_at_threshold)
        tpr.append(tpr_at_threshold)
        fpr.append(fpr_at_threshold)

    return tpr, fpr, thresholds


def precision_recall_curve():
    pass


def auc():
    pass
