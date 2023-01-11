from dataclasses import fields

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import minitorch.autodiff.tensor_functions as tf
import minitorch.datasets as data
from minitorch.autodiff import FastBackend
from minitorch.module import TensorNetwork
from minitorch.optim import SGDOptimizer
from minitorch.tensor_losses import binary_cross_entropy
from minitorch.tensor_metrics import roc_curve
from minitorch.tensor_plotting import plot_tensor_predictions

# Configure page
st.set_page_config(
    page_title="Binary Classifiers",
    page_icon=":white_check_mark:",
    layout="wide",
)
st.title("Binary classifier")

# Configure sidebar
dataset_type = st.sidebar.radio(
    "Dataset type:",
    [field.name.capitalize() for field in fields(data.Datasets)],
)
dataset_size = st.sidebar.slider(
    "Dataset size:",
    min_value=50,
    max_value=200,
    value=100,
)

# Configure network and training process
with st.form("training_config"):
    st.subheader("**Configure network and optimizer**")
    hidden_dims = st.slider(
        "**Number of hidden dimensions**",
        min_value=2,
        max_value=20,
        value=10,
    )
    learning_rate = st.slider(
        "**Learning rate**",
        min_value=1e-03,
        max_value=3.0,
        value=0.3,
    )
    n_epochs = st.slider(
        "**Number of epochs**",
        min_value=10,
        max_value=200,
        value=100,
    )
    train_classifier = st.form_submit_button("Train classifier")

# Initialise data, network and optimizer
datasets = data.Datasets.generate_datasets(dataset_size)
dataset = datasets[dataset_type.lower()]
features = tf.tensor([list(x) for x in dataset.xs], backend=FastBackend)
labels = tf.tensor(dataset.ys, backend=FastBackend).view(dataset_size, 1)
network = TensorNetwork(
    input_dim=2,
    hidden_dim=hidden_dims,
    output_dim=1,
    backend=FastBackend,
)
optimizer = SGDOptimizer(network.parameters(), lr=learning_rate)


# Train classifier
if train_classifier:
    col1, col2, col3 = st.columns(3)
    with col1:
        all_losses = []
        with st.empty():
            for epoch in range(1, n_epochs + 1):
                # Take one step
                optimizer.zero_grad()
                y_hat = network.forward(features).sigmoid()
                loss_per_epoch = binary_cross_entropy(labels, y_hat)
                loss_per_epoch.backward()
                optimizer.step()

                # Update training
                all_losses.append(loss_per_epoch.item())
                fig, ax = plt.subplots()
                ax.plot(
                    list(range(epoch)),
                    all_losses,
                    color=matplotlib.cm.get_cmap("Reds")(0.75),
                    marker="o",
                    markersize=5,
                    lw=2,
                )
                ax.set_title(f"Loss vs. epoch for {dataset_type} dataset")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("BCE Loss")
                ax.grid("on", ls="--")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    with col2:
        # Plot roc curve
        y_hat = network.forward(features).sigmoid().view(dataset_size)
        labels = labels.view(dataset_size)
        tpr, fpr, _ = roc_curve(labels, y_hat)
        fig, ax = plt.subplots(1, 1)
        ax.plot(
            np.array(fpr.data.storage),
            np.array(tpr.data.storage),
            color=matplotlib.cm.get_cmap("Reds")(0.75),
            marker="o",
            markersize=5,
            lw=2,
        )
        ax.plot([0, 1], [0, 1], ls="--", c="tab:grey")
        ax.set_title(f"ROC curve for {dataset_type} dataset")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks(np.linspace(-0.1, 1.1, 13))
        ax.grid("on", ls="--")
        fig.tight_layout()
        st.pyplot(fig)

    with col3:
        # Plot tensor predictions
        fig = plot_tensor_predictions(
            dataset,
            network,
            matplotlib.cm.get_cmap("Blues")(0.75),
            matplotlib.cm.get_cmap("Reds")(0.75),
        )
        st.pyplot(fig)
