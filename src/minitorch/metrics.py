from typing import List

from minitorch.autodiff import Scalar
from minitorch.functional import summation
from minitorch.utils import check_for_binary_values, check_for_equal_dim


def accuracy(y_true: List[Scalar], y_hat: List[Scalar]) -> Scalar:
    check_for_equal_dim(y_true, y_hat)
    check_for_binary_values(y_true)
    check_for_binary_values(y_hat)
    n_correct = summation([t == p for (t, p) in zip(y_true, y_hat)])
    return n_correct / len(y_true)


def true_positive_rate(y_true: List[Scalar], y_hat: List[Scalar]) -> Scalar:
    check_for_equal_dim(y_true, y_hat)
    check_for_binary_values(y_true)
    check_for_binary_values(y_hat)
    tp = summation([(t == p) and (t == 1) for (t, p) in zip(y_true, y_hat)])
    fn = summation([(t != p) and (t == 1) for (t, p) in zip(y_true, y_hat)])
    return tp / (tp + fn)


def false_positive_rate():
    pass


def precision():
    pass


def recall(y_true: List[Scalar], y_hat: List[Scalar]) -> Scalar:
    return true_positive_rate(y_true, y_hat)


def sensitivity(y_true: List[Scalar], y_hat: List[Scalar]) -> Scalar:
    return true_positive_rate(y_true, y_hat)


def specificity():
    return


def roc_curve():
    pass


def precision_recall_curve():
    pass


def auc():
    pass
