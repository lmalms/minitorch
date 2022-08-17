import random

from minitorch.autodiff import Scalar
from minitorch.losses import mean_squared_error
from minitorch.operators import is_close


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
    pass
