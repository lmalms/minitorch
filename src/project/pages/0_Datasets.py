from dataclasses import asdict

import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Polygon, Rectangle

from minitorch.datasets import (
    Dataset,
    Datasets,
    DatasetSplit,
    DatasetTypes,
    Features,
    Labels,
)

# Configure page
st.set_page_config(page_title="Datasets", page_icon=":chart_with_upwards_trend:")
st.title("Datasets")

# Configure side bar
# TODO: How can change the settings (font, size) in the sidebar?
dataset_size = st.sidebar.slider(
    "Dataset size:",
    min_value=50,
    max_value=500,
    value=200,
)
dataset_type = st.sidebar.radio(
    "Dataset type:",
    [
        name.capitalize() if value != DatasetTypes.xor else name.upper()
        for (name, value) in DatasetTypes.__members__.items()
    ],
)

# Initialise dataset
datasets = Datasets.generate_datasets(dataset_size)

# TODO: add a heading above the actual plot.


def select_samples_by_label(dataset: Dataset, label: float):
    features = [(x1, x2) for ((x1, x2), y) in zip(dataset.xs, dataset.ys) if y == label]
    labels = [y for y in dataset.ys if y == label]
    return features, labels


def split_binary_classes(dataset_type: DatasetTypes) -> DatasetSplit:
    dataset = asdict(datasets)[dataset_type]

    # Select features and labels for y == 1.
    features_pos_class, labels_pos_class = select_samples_by_label(dataset, label=1.0)
    features_neg_class, labels_neg_class = select_samples_by_label(dataset, label=0.0)

    return features_pos_class, features_neg_class, labels_pos_class, labels_neg_class


def plot_dataset_split(features: Features, labels: Labels, axis, color, marker):
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
    dataset_type=DatasetTypes(dataset_type.lower())
)

# Plot
fig, ax = plt.subplots(1, 1)

plot_dataset_split(pos_features, pos_labels, ax, "tab:red", "o")
plot_dataset_split(neg_features, neg_labels, ax, "tab:blue", "x")

format_axes(ax)

if dataset_type.lower() == DatasetTypes.simple:
    bottom_left = Rectangle((0, 0), 0.5, 1.0, color="tab:red", alpha=0.2, lw=0.0)
    react2 = Rectangle((0.5, 0), 0.5, 1.0, color="tab:blue", alpha=0.2, lw=0.0)
    ax.add_patch(bottom_left)
    ax.add_patch(react2)
elif dataset_type.lower() == DatasetTypes.split:
    react1 = Rectangle((0, 0), 0.2, 1.0, color="tab:red", alpha=0.2, lw=0.0)
    react2 = Rectangle((0.2, 0), 0.6, 1.0, color="tab:blue", alpha=0.2, lw=0.0)
    react3 = Rectangle((0.8, 0), 0.2, 1.0, color="tab:red", alpha=0.2, lw=0.0)
    ax.add_patch(react1)
    ax.add_patch(react2)
    ax.add_patch(react3)
elif dataset_type.lower() == DatasetTypes.diagonal:
    bottom_left = Polygon([[0, 1], [1, 0], [0, 0]], color="tab:red", alpha=0.2, lw=0.0)
    top_right = Polygon([[0, 1], [1, 1], [1, 0]], color="tab:blue", alpha=0.2, lw=0.0)
    ax.add_patch(bottom_left)
    ax.add_patch(top_right)
elif dataset_type.lower() == DatasetTypes.xor:
    bottom_left = Rectangle((0, 0), 0.5, 0.5, color="tab:blue", alpha=0.2, lw=0.0)
    top_right = Rectangle((0.5, 0.5), 0.5, 0.5, color="tab:blue", alpha=0.2, lw=0.0)
    top_left = Rectangle((0.0, 0.5), 0.5, 0.5, color="tab:red", alpha=0.2, lw=0.0)
    bottom_right = Rectangle((0.5, 0.0), 0.5, 0.5, color="tab:red", alpha=0.2, lw=0.0)
    ax.add_patch(bottom_left)
    ax.add_patch(bottom_right)
    ax.add_patch(top_left)
    ax.add_patch(top_right)
st.pyplot(fig)
