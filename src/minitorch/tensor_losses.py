from minitorch.autodiff import Tensor


def _check_dims(y_true: Tensor, y_hat: Tensor):
    assert y_true.shape == y_hat.shape, "tensors need to have the same shape."
    assert y_true.dims == y_hat.dims == 2, "tensor should be 2D (n_samples, 1)."
    assert y_true.shape[1] == y_hat.shape[1] == 1, "second dimension should equal 1."


def mean_squared_error(y_true: Tensor, y_hat: Tensor) -> Tensor:
    _check_dims(y_true, y_hat)
    return (y_true - y_hat).square().mean()


def binary_cross_entropy(y_true: Tensor, y_hat: Tensor) -> Tensor:
    _check_dims(y_true, y_hat)
    log_likelihoods = (y_true == 1) * y_hat.log() + (y_true == 0) * (-y_hat + 1).log()
    cross_entropy = -log_likelihoods.sum() / float(y_true.shape[0])
    return cross_entropy
