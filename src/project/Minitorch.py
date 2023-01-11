import streamlit as st

st.set_page_config(page_title="minitorch", layout="wide")
st.title("Minitorch")
st.subheader("About this project")
st.text_area(
    label="About this app",
    height=130,
    value="The minitorch project is a DIY machine learning project that implements "
    "some of the core functionality for training neural networks (such as "
    "autodifferentiation, backpropagation and tensor operations) from scratch "
    "in native Python. This app lets user experiment with training neural "
    "networks for a binary classification task on a range of different datasets. "
    "The different datasets can be explored and visualised under the Datasets page, "
    "and classifiers can be trained on the Binary Classifier page.",
    label_visibility="collapsed",
)
