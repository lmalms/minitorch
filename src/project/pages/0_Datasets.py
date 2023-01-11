from dataclasses import fields

import streamlit as st

from minitorch.datasets import Datasets

# Configure page
st.set_page_config(
    page_title="Datasets",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)
st.title("Datasets")

# About this page
with st.expander("About this page"):
    st.write("....")

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
    [field.name.capitalize() for field in fields(Datasets)],
)

# Initialise and plot dataset
datasets = Datasets.generate_datasets(dataset_size)
fig = datasets[dataset_type.lower()].plot()
st.pyplot(fig)
