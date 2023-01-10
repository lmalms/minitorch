from dataclasses import fields

import matplotlib
import numpy as np
import streamlit as st

import minitorch.autodiff.tensor_functions as tf
import minitorch.datasets as data
from minitorch.autodiff import FastBackend, Tensor
from minitorch.module import TensorNetwork
from minitorch.optim import SGDOptimizer
from minitorch.tensor_losses import binary_cross_entropy
from minitorch.tensor_plotting import plot_tensor_predictions


###
# Check out animation demo in $streamlit hello - might be some useful stuff in there
###


# Utils for this page
datasets = {
    "simple": data.SimpleDataset,
    "diagonal": data.DiagonalDataset,
    "split": data.SplitDataset,
    "xor": data.XORDataset,
}


def train_classifier(
    network: TensorNetwork,
    features: Tensor,
    labels: Tensor,
    optimizer: SGDOptimizer,
    n_epochs: int,
) -> None:

    all_losses = []
    latest_step = st.empty()
    progress_bar = st.progress(0)

    for epoch in range(1, n_epochs + 1):
        # Take on step
        optimizer.zero_grad()
        y_hat = network.forward(features).sigmoid()
        loss_per_epoch = binary_cross_entropy(labels, y_hat)
        loss_per_epoch.backward()
        optimizer.step()

        # Record
        all_losses.append(loss_per_epoch.item())
        latest_step.text(f"Epoch {epoch}: Loss = {loss_per_epoch.item():.3f}")
        progress_bar.progress(epoch)


def plot_predictions(network: TensorNetwork, dataset: data.Dataset):
    fig = plot_tensor_predictions(
        dataset,
        network,
        matplotlib.cm.get_cmap("Blues")(0.5),
        matplotlib.cm.get_cmap("Reds")(0.5),
    )
    st.pyplot(fig)


# Configure page
st.set_page_config(page_title="Binary Classifiers", page_icon=":white_check_mark:")
st.title("Train a binary classifier")

# Configure side bar
with st.sidebar:

    # Configure dataset
    st.text("1. Configure dataset")
    dataset_type = st.radio(
        "Dataset type:",
        [field.name.capitalize() for field in fields(data.Datasets)],
    )
    dataset_size = st.slider(
        "Dataset size:",
        min_value=50,
        max_value=200,
        value=100,
    )

    # Configure network
    st.text("2. Configure network")
    hidden_dims = st.slider(
        "Number of hidden dimensions",
        min_value=2,
        max_value=20,
        value=10,
    )

    # Configure training process
    st.text("3. Configure optimizer")
    learning_rate = st.slider(
        "Learning rate",
        min_value=1e-03,
        max_value=3.0,
        value=0.3,
    )
    n_epochs = st.slider(
        "Number of epochs",
        min_value=10,
        max_value=200,
        value=100,
    )


# Initialise data, network and optimizer
dataset = datasets[dataset_type.lower()](dataset_size)
X = tf.tensor([list(x) for x in dataset.xs], backend=FastBackend)
y_true = tf.tensor(dataset.ys, backend=FastBackend).view(dataset_size, 1)
network = TensorNetwork(
    input_dim=2,
    hidden_dim=hidden_dims,
    output_dim=1,
    backend=FastBackend,
)
optimizer = SGDOptimizer(network.parameters(), lr=learning_rate)


# Train
st.button(
    label="Train classifier",
    on_click=train_classifier,
    args=(network, X, y_true, optimizer, n_epochs),
)

# Plot predictions
st.button(
    label="Plot predictions",
    on_click=plot_predictions,
    args=(network, dataset),
)
