from minitorch.autodiff import Tensor


def _check_dims(y_true: Tensor, y_hat: Tensor):
    assert y_true.shape == y_hat.shape, "tensors need to have the same shape."
    assert y_true.dims == y_hat.dims == 1, "tensor should be 1D (n_samples, )."


def _check_for_binary_values(y: Tensor):
    def is_binary(y: Tensor) -> bool:
        all_binary = ((y == 0.0) + (y == 1.0)).all().item()
        return bool(all_binary)

    assert is_binary(y), "y can only contain values of 1. or 0."
