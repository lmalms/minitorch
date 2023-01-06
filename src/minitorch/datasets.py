from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

# Type hints
Features = List[Tuple[float, float]]
Labels = List[float]
DatasetSplit = Tuple[Features, Features, Labels, Labels]


class Dataset:
    def __init__(self, n: int):
        self.n = n
        self._rng = np.random.default_rng()
        self._xs = self._generate_xs()
        self._ys = self._generate_ys()

    @property
    def xs(self) -> Features:
        return self._xs

    @property
    def ys(self) -> Labels:
        return self._ys

    @property
    @abstractmethod
    def type(self) -> str:
        ...

    def _generate_xs(self) -> Features:
        return [(self._rng.uniform(), self._rng.uniform()) for _ in range(self.n)]

    @abstractmethod
    def _generate_ys(self) -> Labels:
        ...

    @classmethod
    def generate(cls, n: int) -> Dataset:
        return Dataset(n)

    def split_by_class(self) -> DatasetSplit:
        positive_split = [(x, y) for (x, y) in zip(self.xs, self.ys) if y == 1]
        negative_split = [(x, y) for (x, y) in zip(self.xs, self.ys) if y == 0]

        # Split out features and labels
        positive_features, positive_labels = zip(*positive_split)
        negative_features, negative_labels = zip(*negative_split)

        return (
            positive_features,
            negative_features,
            positive_labels,
            negative_labels,
        )

    def plot(self):
        # Split dataset
        positive_features, negative_features, _, _ = self.split_by_class()
        positive_x1, positive_x2 = zip(*positive_features)
        negative_x1, negative_x2 = zip(*negative_features)

        # Plot dataset
        fig, ax = plt.subplots(1, 1, dpi=110)
        ax.scatter(
            list(positive_x1),
            list(positive_x2),
            marker="x",
            c="tab:blue",
            label="class = 1",
        )
        ax.scatter(
            list(negative_x1),
            list(negative_x2),
            marker="o",
            c="tab:red",
            label="class = 0",
        )
        ax.legend(loc=1)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_ylim(-0.05, 1.05)
        fig.tight_layout()

        return fig


class SimpleDataset(Dataset):
    def __init__(self, n: int):
        super().__init__(n)

    @property
    def type(self) -> str:
        return "simple"

    def _generate_ys(self) -> Labels:
        return [1 if x1 < 0.5 else 0.0 for (x1, _) in self.xs]

    def plot(self, add_shading: bool = True):
        fig = super().plot()
        ax = plt.gca()

        if add_shading:
            # Add patches to highlight positive and negative class regiongs
            left = Rectangle((0, 0), 0.5, 1.0, color="tab:blue", alpha=0.2, lw=0.0)
            right = Rectangle((0.5, 0), 0.5, 1.0, color="tab:red", alpha=0.2, lw=0.0)
            ax.add_patch(left)
            ax.add_patch(right)

        ax.set_title("Simple Dataset")
        fig.tight_layout()

        return fig


class DiagonalDataset(Dataset):
    def __init__(self, n: int):
        super().__init__(n)

    @property
    def type(self) -> str:
        return "diagonal"

    def _generate_ys(self) -> Labels:
        return [1 if (x1 + x2 < 1.0) else 0.0 for (x1, x2) in self.xs]

    def plot(self, add_shading: bool = True):
        fig = super().plot()
        ax = plt.gca()

        if add_shading:
            # Add patches to highlight positive and negative class regiongs
            left = Polygon([[0, 1], [1, 0], [0, 0]], color="tab:blue", alpha=0.2, lw=0)
            right = Polygon([[0, 1], [1, 1], [1, 0]], color="tab:red", alpha=0.2, lw=0)
            ax.add_patch(left)
            ax.add_patch(right)

        ax.set_title("Diagonal Dataset")
        fig.tight_layout()

        return fig


class SplitDataset(Dataset):
    def __init__(self, n: int):
        super().__init__(n)

    @property
    def type(self) -> str:
        return "split"

    def _generate_ys(self) -> Labels:
        return [1 if (x1 < 0.2) or (x1 > 0.8) else 0.0 for (x1, _) in self.xs]

    def plot(self, add_shading: bool = True):
        fig = super().plot()
        ax = plt.gca()

        if add_shading:
            # Add patches to highlight positive and negative class regiongs
            left = Rectangle((0.0, 0.0), 0.2, 1.0, color="tab:blue", alpha=0.2, lw=0.0)
            center = Rectangle((0.2, 0.0), 0.6, 1.0, color="tab:red", alpha=0.2, lw=0.0)
            right = Rectangle((0.8, 0.0), 0.2, 1.0, color="tab:blue", alpha=0.2, lw=0.0)
            ax.add_patch(left)
            ax.add_patch(center)
            ax.add_patch(right)

        ax.set_title("Split Dataset")
        fig.tight_layout()

        return fig


class XORDataset(Dataset):
    def __init__(self, n: int):
        super().__init__(n)

    @property
    def type(self) -> str:
        return "xor"

    def _generate_ys(self) -> Labels:
        return [
            1 if ((x1 < 0.5) and (x2 > 0.5)) or ((x1 > 0.5) and (x2 < 0.5)) else 0.0
            for (x1, x2) in self.xs
        ]

    def plot(self, add_shading: bool = True):
        fig = super().plot()
        ax = plt.gca()

        if add_shading:
            # Add patches to highlight positive and negative class regiongs
            bl = Rectangle((0, 0), 0.5, 0.5, color="tab:red", alpha=0.2, lw=0.0)
            tr = Rectangle((0.5, 0.5), 0.5, 0.5, color="tab:red", alpha=0.2, lw=0.0)
            tl = Rectangle((0.0, 0.5), 0.5, 0.5, color="tab:blue", alpha=0.2, lw=0.0)
            br = Rectangle((0.5, 0.0), 0.5, 0.5, color="tab:blue", alpha=0.2, lw=0.0)

            ax.add_patch(bl)
            ax.add_patch(tr)
            ax.add_patch(tl)
            ax.add_patch(br)

        ax.set_title("XOR Dataset")
        fig.tight_layout()

        return fig


@dataclass
class Datasets:
    """
    Container for storing all datasets.
    """

    simple: SimpleDataset
    diagonal: DiagonalDataset
    split: SplitDataset
    xor: XORDataset

    @property
    def all_datasets(self) -> List[Dataset]:
        return [self.simple, self.diagonal, self.split, self.xor]

    @property
    def dataset_types(self) -> List[str]:
        return [dataset.type for dataset in self.all_datasets]

    @classmethod
    def generate_datasets(cls, n_samples: int = 200) -> Datasets:
        return Datasets(
            simple=SimpleDataset(n_samples),
            split=SplitDataset(n_samples),
            diagonal=DiagonalDataset(n_samples),
            xor=XORDataset(n_samples),
        )

    def __getitem__(self, dataset: str) -> Dataset:
        if dataset not in self.dataset_types:
            raise ValueError(
                f"Dataset type {dataset} is not one of supported datasets. "
                f"Supported datasets are: {self.dataset_types}."
            )

        return self.__dict__[dataset]
