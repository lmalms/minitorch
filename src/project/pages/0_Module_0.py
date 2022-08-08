from typing import List, Tuple

import matplotlib.pyplot as plt
import streamlit as st
from datasets import DiagonalDataset, SimpleDataset, SplitDataset, XORDataset
from matplotlib.patches import Rectangle

# App configuration
N = 200
DATASETS = {
    "simple": SimpleDataset(N),
    "split": SplitDataset(N),
    "diagonal": DiagonalDataset(N),
    "xor": XORDataset(N),
}


# App
"""
Module 0
"""

selected_dataset = st.selectbox(
    "Choose dataset:", ["Simple", "Split", "Diagonal", "XOR"]
)


def split_binary_classes(
    dataset: str,
) -> Tuple[
    List[Tuple[float, float]], List[Tuple[float, float]], List[float], List[float]
]:

    # Select features and labels from y == 1.
    features_pos_class = [
        (x1, x2)
        for ((x1, x2), y) in zip(DATASETS[dataset].xs, DATASETS[dataset].ys)
        if y == 1.0
    ]
    labels_pos_class = [y for y in DATASETS[dataset].ys if y == 1.0]

    # Select features and labels from y == 0.
    features_neg_class = [
        (x1, x2)
        for ((x1, x2), y) in zip(DATASETS[dataset].xs, DATASETS[dataset].ys)
        if y == 0.0
    ]
    labels_neg_class = [y for y in DATASETS[dataset].ys if y == 0.0]

    return (features_pos_class, features_neg_class, labels_pos_class, labels_neg_class)


def plot_dataset_split(
    features: List[Tuple[float, float]], labels: List[float], axis, color, marker
):
    # Plot a dataset split on a matplotlib axis object.
    x1, x2 = zip(*features)
    axis.scatter(
        x1,
        x2,
        c=color,
        marker=marker,
        s=50,
    )
    return axis


def format_axes(ax):
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    return ax


# Split out features and labels by class
pos_features, neg_features, pos_labels, neg_labels = split_binary_classes(
    dataset=selected_dataset.lower()
)

# Plot
fig, ax = plt.subplots(1, 1)

plot_dataset_split(pos_features, pos_labels, ax, "tab:red", "o")
plot_dataset_split(neg_features, neg_labels, ax, "tab:blue", "x")

format_axes(ax)

if selected_dataset.lower() == "simple":
    bottom_left = Rectangle((0, 0), 0.5, 1.0, color="tab:red", alpha=0.3, lw=0.0)
    react2 = Rectangle((0.5, 0), 0.5, 1.0, color="tab:blue", alpha=0.3, lw=0.0)
    ax.add_patch(bottom_left)
    ax.add_patch(react2)
elif selected_dataset.lower() == "split":
    react1 = Rectangle((0, 0), 0.2, 1.0, color="tab:red", alpha=0.3, lw=0.0)
    react2 = Rectangle((0.2, 0), 0.6, 1.0, color="tab:blue", alpha=0.3, lw=0.0)
    react3 = Rectangle((0.8, 0), 0.2, 1.0, color="tab:red", alpha=0.3, lw=0.0)
    ax.add_patch(react1)
    ax.add_patch(react2)
    ax.add_patch(react3)
elif selected_dataset.lower() == "xor":
    bottom_left = Rectangle((0, 0), 0.5, 0.5, color="tab:blue", alpha=0.3, lw=0.0)
    top_right = Rectangle((0.5, 0.5), 0.5, 0.5, color="tab:blue", alpha=0.3, lw=0.0)
    top_left = Rectangle((0.0, 0.5), 0.5, 0.5, color="tab:red", alpha=0.3, lw=0.0)
    bottom_right = Rectangle((0.5, 0.0), 0.5, 0.5, color="tab:red", alpha=0.3, lw=0.0)
    ax.add_patch(bottom_left)
    ax.add_patch(bottom_right)
    ax.add_patch(top_left)
    ax.add_patch(top_right)
st.pyplot(fig)
