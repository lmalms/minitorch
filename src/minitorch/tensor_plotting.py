from typing import Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from minitorch.autodiff import tensor
from minitorch.datasets import Dataset
from minitorch.module import Module

# Colors
Blue = matplotlib.cm.get_cmap("Blues")(0.75)
Red = matplotlib.cm.get_cmap("Reds")(0.75)


def _add_simple_tensor_predictions(
    ax: Axes,
    model: Module,
    color_positive_class: Union[Tuple[float], str],
    color_negative_class: Union[Tuple[float], str],
) -> Axes:
    x1_res, x2_res = 100, 10
    x1_positions = list(np.linspace(0, 1, (x1_res + 1)))

    for x1_lower, x1_upper in zip(x1_positions, x1_positions[1:]):

        # Create corresponding x2 values for lower
        X_lower = [list((x1_lower, i / x2_res)) for i in range(x2_res + 1)]
        y_lower = model.forward(tensor(X_lower))
        y_mean_lower = y_lower.view(y_lower.size).mean().sigmoid()

        # Create corresponding x2 values for upper
        X_upper = [list((x1_upper, i / x2_res)) for i in range(x2_res + 1)]
        y_upper = model.forward(tensor(X_upper))
        y_mean_upper = y_upper.view(y_upper.size).mean().sigmoid()

        # Average
        y_mean = ((y_mean_upper + y_mean_lower) / 2).item()

        # Plot and fill
        ax.fill_betweenx(
            [i / x2_res for i in range(x2_res + 1)],
            [x1_upper for _ in range(x2_res + 1)],
            [x1_lower for _ in range(x2_res + 1)],
            alpha=(y_mean - 0.5) if y_mean >= 0.5 else (0.5 - y_mean),
            color=color_positive_class if y_mean >= 0.5 else color_negative_class,
            lw=0.001,
        )

    return ax


def _add_diagonal_tensor_predictions(
    ax: Axes,
    model: Module,
    color_positive_class: Union[Tuple[float], str],
    color_negative_class: Union[Tuple[float], str],
) -> Axes:
    x1_res, bias_res = 100, 100
    x_positions = list(np.linspace(0, 1, x1_res + 1))
    bias_positions = list(np.linspace(0, 2, bias_res + 1))

    for bias_lower, bias_upper in zip(bias_positions, bias_positions[1:]):

        # Create corresponding x2 values for lower
        X_lower = [list((x1, bias_lower - x1)) for x1 in x_positions]

        # Clamp values to between 0.0 and 1.0
        X_lower = [[x1, max(min(x2, 1.0), 0.0)] for (x1, x2) in X_lower]
        y_lower = model.forward(tensor(X_lower))
        y_mean_lower = y_lower.view(y_lower.size).mean().sigmoid()

        # Create corresponding x2 values for upper and clamp
        X_upper = [list((x1, bias_upper - x1)) for x1 in x_positions]
        X_upper = [[x1, max(min(x2, 1.0), 0.0)] for (x1, x2) in X_upper]
        y_upper = model.forward(tensor(X_upper))
        y_mean_upper = y_upper.view(y_upper.size).mean().sigmoid()

        # Average
        y_mean = ((y_mean_upper + y_mean_lower) / 2).item()

        # Plot and fill
        ax.fill_between(
            x_positions,
            [x2 for (_, x2) in X_lower],
            [x2 for (_, x2) in X_upper],
            alpha=(y_mean - 0.5) if y_mean >= 0.5 else (0.5 - y_mean),
            color=color_positive_class if y_mean >= 0.5 else color_negative_class,
            lw=0.001,
        )

    return ax


def _add_split_tensor_predictions(
    ax: Axes,
    model: Module,
    color_positive_class: Union[Tuple[float], str],
    color_negative_class: Union[Tuple[float], str],
) -> Axes:
    x1_res, x2_res = 100, 10
    x1_positions = list(np.linspace(0, 1, x1_res + 1))
    x2_positions = list(np.linspace(0, 1, x2_res + 1))

    for x1_lower, x1_upper in zip(x1_positions, x1_positions[1:]):

        # Create corresponding x2 values
        X_lower = [list((x1_lower, x2)) for x2 in x2_positions]
        X_upper = [list((x1_upper, x2)) for x2 in x2_positions]

        # Get predictions
        y_lower = model.forward(tensor(X_lower))
        y_mean_lower = y_lower.view(y_lower.size).mean().sigmoid()

        y_upper = model.forward(tensor(X_upper))
        y_mean_upper = y_upper.view(y_upper.size).mean().sigmoid()

        # Avergage
        y_mean = ((y_mean_upper + y_mean_lower) / 2).item()

        # Plot and fill
        ax.fill_betweenx(
            x2_positions,
            [x1_upper for _ in x2_positions],
            [x1_lower for _ in x2_positions],
            alpha=(y_mean - 0.5) if y_mean >= 0.5 else (0.5 - y_mean),
            color=color_positive_class if y_mean >= 0.5 else color_negative_class,
            lw=0.001,
        )

    return ax


def _add_xor_tensor_predictions(
    ax: Axes,
    model: Module,
    color_positive_class: Union[Tuple[float], str],
    color_negative_class: Union[Tuple[float], str],
) -> Axes:
    x1_res, x2_res = 100, 100
    x1_positions = list(np.linspace(0, 1.0, x1_res))
    x2_positions = list(np.linspace(0, 1.0, x2_res))

    # Generate predictions for each point on mesh
    y_hat = np.zeros(shape=(x1_res, x2_res))
    for i, x1 in enumerate(x1_positions):
        X = [[x1, x2] for x2 in x2_positions]
        y_hat_ = model.forward(tensor(X))
        y_hat_ = y_hat_.sigmoid()
        y_hat[:, i] = y_hat_.data.storage.flatten()

    # Plot predictions as rectangles
    for i, (x1_lower, x1_upper) in enumerate(zip(x1_positions, x1_positions[1:])):
        for j, (x2_lower, x2_upper) in enumerate(zip(x2_positions, x2_positions[1:])):
            color = color_positive_class if y_hat[i, j] >= 0.5 else color_negative_class
            alpha = (y_hat[i, j] - 0.5) if y_hat[i, j] >= 0.5 else (0.5 - y_hat[i, j])
            rect = Rectangle(
                (x1_lower, x2_lower),
                (x1_upper - x1_lower),
                (x2_upper - x2_lower),
                color=color,
                alpha=alpha,
                lw=0.0,
            )
            ax.add_patch(rect)

    return ax


def plot_tensor_predictions(
    dataset: Dataset,
    model: Module,
    color_positive_class: Union[Tuple[float], str] = Blue,
    color_negative_class: Union[Tuple[float], str] = Red,
):
    fig = dataset.plot(add_shading=False)
    ax = plt.gca()

    if dataset.type == "simple":
        ax = _add_simple_tensor_predictions(
            ax,
            model,
            color_positive_class,
            color_negative_class,
        )

    elif dataset.type == "diagonal":
        ax = _add_diagonal_tensor_predictions(
            ax,
            model,
            color_positive_class,
            color_negative_class,
        )

    elif dataset.type == "split":
        ax = _add_split_tensor_predictions(
            ax,
            model,
            color_positive_class,
            color_negative_class,
        )

    elif dataset.type == "xor":
        ax = _add_xor_tensor_predictions(
            ax,
            model,
            color_positive_class,
            color_negative_class,
        )

    fig.tight_layout()
    return fig
