import random

import numpy as np

import minitorch.autodiff.tensor_functions as tf
from minitorch.operators import log
from minitorch.tensor_losses import binary_cross_entropy, mean_squared_error


def test_mse_loss():
    n_samples = 10

    # Predicting the target exactly should have zero error
    y_true = tf.rand((n_samples, 1))
    mse = mean_squared_error(y_true, y_true)
    assert np.all(np.array([0.0]) == mse.data.storage)

    # Check using operators
    y_true, y_hat = tf.rand((n_samples, 1)), tf.rand((n_samples, 1))
    mse = mean_squared_error(y_true, y_hat)

    mse_check = 0
    for y_t, y_h in zip(y_true.data.storage, y_hat.data.storage):
        mse_check += (y_t - y_h) ** 2

    mse_check = mse_check / n_samples
    assert np.allclose(np.array([mse_check]), mse.data.storage)


def test_bce_loss():
    n_samples = 10

    # Check with operator implementation
    y_true = tf.tensor([[random.randint(0, 1)] for _ in range(n_samples)])
    y_hat = tf.rand((n_samples, 1))
    bce = binary_cross_entropy(y_true, y_hat)

    bce_check = 0
    for (y_t, y_h) in zip(y_true.data.storage, y_hat.data.storage):
        if y_t == 1:
            bce_check += log(y_h)
        elif y_t == 0:
            bce_check += log(1 - y_h)

    bce_check = -bce_check / n_samples
    assert np.allclose(np.array([bce_check]), bce.data.storage)
