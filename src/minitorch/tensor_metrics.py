from minitorch.autodiff import Tensor


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
