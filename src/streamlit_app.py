import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from minitorch.datasets import *

"""
Module 0
"""

n = 200

datasets = {
    "simple": SimpleDataset(n),
    "split": SplitDataset(n),
    "diagonal": DiagonalDataset(n),
    "xor": XORDataset(n)
}

dataset = st.selectbox(
    "Choose dataset:",
    [
        "Simple",
        "Split",
        "Diagonal",
        "XOR"
    ]
)

x1 = [x1 for (x1, _) in datasets[dataset.lower()].xs]
x2 = [x2 for (_, x2) in datasets[dataset.lower()].xs]
ys = [y for y in datasets[dataset.lower()].ys]

fig, ax = plt.subplots(1, 1)
ax.scatter(
    x1,
    x2,
    c=["tab:red" if y == 1. else "tab:blue" for y in ys]
)
ax.set_xlim(0.0, 1.)
ax.set_ylim(0.0, 1.)
ax.set_xlabel("x1")
ax.set_ylabel("x2")

if dataset.lower() == "simple":
    bottom_left = Rectangle(
        (0, 0), 0.5, 1.0, color="tab:red", alpha=0.3, lw=0.
    )
    react2 = Rectangle(
        (0.5, 0), 0.5, 1.0, color="tab:blue", alpha=0.3, lw=0.
    )
    ax.add_patch(bottom_left)
    ax.add_patch(react2)
elif dataset.lower() == "split":
    react1 = Rectangle(
        (0, 0), 0.2, 1.0, color="tab:red", alpha=0.3, lw=0.
    )
    react2 = Rectangle(
        (0.2, 0), 0.6, 1.0, color="tab:blue", alpha=0.3, lw=0.
    )
    react3 = Rectangle(
        (0.8, 0), 0.2, 1.0, color="tab:red", alpha=0.3, lw=0.
    )
    ax.add_patch(react1)
    ax.add_patch(react2)
    ax.add_patch(react3)
elif dataset.lower() == "xor":
    bottom_left = Rectangle(
        (0, 0), 0.5, 0.5, color="tab:blue", alpha=0.3, lw=0.
    )
    top_right = Rectangle(
        (0.5, 0.5), 0.5, 0.5, color="tab:blue", alpha=0.3, lw=0.
    )
    top_left = Rectangle(
        (0., 0.5), 0.5, 0.5, color="tab:red", alpha=0.3, lw=0.
    )
    bottom_right = Rectangle(
        (0.5, 0.), 0.5, 0.5, color="tab:red", alpha=0.3, lw=0.
    )
    ax.add_patch(bottom_left)
    ax.add_patch(bottom_right)
    ax.add_patch(top_left)
    ax.add_patch(top_right)
st.pyplot(fig)

