from enum import Enum

import streamlit as st

from minitorch.datasets import Dataset, DatasetTypes
from minitorch.module import Linear, Module, Network


# Types and other utils needed for this page
class NetworkType(str, Enum):
    layer = "layer"
    network = "network"


# Configure page
st.set_page_config(page_title="Binary Classifiers", page_icon=":white_check_mark:")
st.title("Training a binary classifier")

# Configure side bar
with st.sidebar:
    st.subheader("1. Configure dataset")
    dataset_type = st.radio(
        "Dataset type:",
        [
            name.capitalize() if value != DatasetTypes.xor else name.upper()
            for (name, value) in DatasetTypes.__members__.items()
        ],
    )
    # TODO: what configs here work well for all systems?
    #  Should I maybe hard code them?
    dataset_size = st.slider("Dataset size:", min_value=50, max_value=200, value=100)

    st.subheader("2. Configure network")
    network = st.radio(
        "Layer vs. Network",
        [name.capitalize() for (name, _) in NetworkType.__members__.items()],
    )

    st.subheader("3. Configure SGD optimizer")
    st.caption("Note: Values of xxx work well.")
    learning_rate = st.slider(
        "Learning rate", min_value=1e-03, max_value=0.75, value=0.375
    )
    n_epochs = st.slider("Number of epochs", min_value=10, max_value=200, value=100)


st.header("Network visualisation to go here ...")


def train(network: Module, dataset: Dataset, learing_rate: float, n_epochs: float):
    return st.success(f"training with learning rate {learing_rate}")


# Train
st.button(
    label="Train", on_click=train, args=(network, dataset_type, learning_rate, n_epochs)
)

#
