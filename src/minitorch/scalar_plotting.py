import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from minitorch.datasets import Dataset
from minitorch.module import Module
from minitorch.operators import sigmoid


def _add_simple_scalar_predictions(ax: Axes, model: Module) -> Axes:
    x1_res, x2_res = 100, 10
    x1_positions = list(np.linspace(0, 1, (x1_res + 1)))

    for x1_lower, x1_upper in zip(x1_positions, x1_positions[1:]):

        # Create corresponding x2 values for lower
        X_lower = [list((x1_lower, i / x2_res)) for i in range(x2_res + 1)]
        y_lower = model.forward(X_lower)
        y_lower = list(itertools.chain.from_iterable(y_lower))
        y_mean_lower = sigmoid(sum(s.data for s in y_lower) / len(y_lower))

        # Create corresponding x2 values for upper
        X_upper = [list((x1_upper, i / x2_res)) for i in range(x2_res + 1)]
        y_upper = model.forward(X_upper)
        y_upper = list(itertools.chain.from_iterable(y_upper))
        y_mean_upper = sigmoid(sum(s.data for s in y_upper) / len(y_upper))

        # Average
        y_mean = (y_mean_upper + y_mean_lower) / 2

        # Plot and fill
        ax.fill_betweenx(
            [i / x2_res for i in range(x2_res + 1)],
            [x1_upper for _ in range(x2_res + 1)],
            [x1_lower for _ in range(x2_res + 1)],
            alpha=(y_mean - 0.5) if y_mean >= 0.5 else (0.5 - y_mean),
            color="tab:blue" if y_mean >= 0.5 else "tab:red",
            lw=0.001,
        )

    return ax


def _add_diagonal_scalar_predictions(ax: Axes, model: Module) -> Axes:
    x1_res, bias_res = 100, 100
    x_positions = list(np.linspace(0, 1, x1_res + 1))
    bias_positions = list(np.linspace(0, 2, bias_res + 1))

    for bias_lower, bias_upper in zip(bias_positions, bias_positions[1:]):

        # Create corresponding x2 values for lower
        X_lower = [list((x1, bias_lower - x1)) for x1 in x_positions]

        # Clamp values to between 0.0 and 1.0
        X_lower = [[x1, max(min(x2, 1.0), 0.0)] for (x1, x2) in X_lower]
        y_lower = model.forward(X_lower)
        y_lower = list(itertools.chain.from_iterable(y_lower))
        y_mean_lower = sigmoid(sum(s.data for s in y_lower) / len(y_lower))

        # Create corresponding x2 values for upper
        X_upper = [list((x1, bias_upper - x1)) for x1 in x_positions]

        # Clamp upper and lower bounds
        X_upper = [[x1, max(min(x2, 1.0), 0.0)] for (x1, x2) in X_upper]
        y_upper = model.forward(X_upper)
        y_upper = list(itertools.chain.from_iterable(y_upper))
        y_mean_upper = sigmoid(sum(s.data for s in y_upper) / len(y_upper))

        # Average
        y_mean = (y_mean_upper + y_mean_lower) / 2

        # Plot and fill
        ax.fill_between(
            x_positions,
            [x2 for (_, x2) in X_lower],
            [x2 for (_, x2) in X_upper],
            alpha=(y_mean - 0.5) if y_mean >= 0.5 else (0.5 - y_mean),
            color="tab:blue" if y_mean >= 0.5 else "tab:red",
            lw=0.001,
        )

    return ax


def _add_split_scalar_predictions(ax: Axes, model: Module) -> Axes:
    x1_res, x2_res = 100, 10
    x1_positions = list(np.linspace(0, 1, x1_res + 1))
    x2_positions = list(np.linspace(0, 1, x2_res + 1))

    for x1_lower, x1_upper in zip(x1_positions, x1_positions[1:]):

        # Create corresponding x2 values
        X_lower = [list((x1_lower, x2)) for x2 in x2_positions]
        X_upper = [list((x1_upper, x2)) for x2 in x2_positions]

        # Get predictions
        y_lower = model.forward(X_lower)
        y_lower = list(itertools.chain.from_iterable(y_lower))
        y_mean_lower = sigmoid(sum(s.data for s in y_lower) / len(y_lower))

        y_upper = model.forward(X_upper)
        y_upper = list(itertools.chain.from_iterable(y_upper))
        y_mean_upper = sigmoid(sum(s.data for s in y_upper) / len(y_upper))

        # Avergage
        y_mean = (y_mean_upper + y_mean_lower) / 2

        # Plot and fill
        ax.fill_betweenx(
            x2_positions,
            [x1_upper for _ in x2_positions],
            [x1_lower for _ in x2_positions],
            alpha=(y_mean - 0.5) if y_mean >= 0.5 else (0.5 - y_mean),
            color="tab:blue" if y_mean >= 0.5 else "tab:red",
            lw=0.001,
        )

    return ax


def _add_xor_scalar_predictions(ax: Axes, model: Module) -> Axes:
    x1_res, x2_res = 100, 100
    x1_positions = list(np.linspace(0, 1.0, x1_res))
    x2_positions = list(np.linspace(0, 1.0, x2_res))

    # Generate predictions for each point on mesh
    y_hat = np.zeros(shape=(x1_res, x2_res))
    for i, x1 in enumerate(x1_positions):
        X = [[x1, x2] for x2 in x2_positions]
        y_hat_ = model.forward(X)
        y_hat_ = [[s.sigmoid().data for s in sample] for sample in y_hat_]
        y_hat[:, i] = np.array(y_hat_).flatten()

    # Plot predictions as rectangles
    for i, (x1_lower, x1_upper) in enumerate(zip(x1_positions, x1_positions[1:])):
        for j, (x2_lower, x2_upper) in enumerate(zip(x2_positions, x2_positions[1:])):
            rect = Rectangle(
                (x1_lower, x2_lower),
                (x1_upper - x1_lower),
                (x2_upper - x2_lower),
                color="tab:blue" if y_hat[i, j] >= 0.5 else "tab:red",
                alpha=(y_hat[i, j] - 0.5)
                if y_hat[i, j] >= 0.5
                else (0.5 - y_hat[i, j]),
                lw=0.0,
            )
            ax.add_patch(rect)

    return ax


def plot_scalar_predictions(
    dataset: Dataset,
    model: Module,
):
    fig = dataset.plot(add_shading=False)
    ax = plt.gca()

    if dataset.type == "simple":
        ax = _add_simple_scalar_predictions(ax, model)

    elif dataset.type == "diagonal":
        ax = _add_diagonal_scalar_predictions(ax, model)

    elif dataset.type == "split":
        ax = _add_split_scalar_predictions(ax, model)

    elif dataset.type == "xor":
        ax = _add_xor_scalar_predictions(ax, model)

    fig.tight_layout()
    return fig
