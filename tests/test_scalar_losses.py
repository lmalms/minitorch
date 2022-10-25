import random

from minitorch.autodiff import Scalar
from minitorch.operators import is_close, log
from minitorch.scalar_losses import binary_cross_entropy, mean_squared_error


def test_mse_loss():
    n_samples = 10

    # Predicting the target exactly should have zero error
    y_true = [Scalar(random.random()) for _ in range(n_samples)]
    mse = mean_squared_error(y_true, y_true)
    assert is_close(0.0, mse.data)

    # Check using operators
    y_true = [Scalar(random.random()) for _ in range(n_samples)]
    y_hat = [Scalar(random.random()) for _ in range(n_samples)]
    mse = mean_squared_error(y_true, y_hat)

    mse_check = 0
    for y_t, y_h in zip(y_true, y_hat):
        mse_check += (y_t.data - y_h.data) ** 2

    mse_check = mse_check / len(y_true)
    assert is_close(mse.data, mse_check)


def test_bce_loss():
    n_samples = 10

    # Predicting the target exactly should have zero error
    y_true = [Scalar(1) for _ in range(n_samples)]
    bce = binary_cross_entropy(y_true, y_true)
    assert is_close(0.0, bce.data)

    # Check with operator implementation
    y_true = [Scalar(random.randint(0, 1)) for _ in range(n_samples)]
    y_hat = [Scalar(random.random()) for _ in range(n_samples)]
    bce = binary_cross_entropy(y_true, y_hat)

    bce_check = 0
    for (y_t, y_h) in zip(y_true, y_hat):
        if y_t == 1:
            bce_check += log(y_h.data)
        elif y_t == 0:
            bce_check += log(1 - y_h.data)

    bce_check = -bce_check / len(y_true)
    assert is_close(bce_check, bce.data)
