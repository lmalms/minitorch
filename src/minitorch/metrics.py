from typing import List

from minitorch.autodiff import Scalar
from minitorch.functional import summation
from minitorch.utils import check_for_equal_dim


def accuracy(y_true: List[Scalar], y_hat: List[Scalar]) -> Scalar:
    check_for_equal_dim(y_true, y_hat)
    n_correct = summation([t == p for (t, p) in zip(y_true, y_hat)])
    return n_correct / len(y_true)


def true_positive_rate():
    pass


def false_positive_rate():
    pass


def precision():
    pass


def recall():
    pass


def roc_curve():
    pass


def precision_recall_curve():
    pass


def auc():
    pass
