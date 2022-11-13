from __future__ import annotations

import random
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


# Type hints
Features = List[Tuple[float, float]]
Labels = List[float]
DatasetSplit = Tuple[Features, Features, Labels, Labels]


class Dataset:

    random.seed(0)

    def __init__(self, n: int):
        self.n = n
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
        return [(random.random(), random.random()) for _ in range(self.n)]

    @abstractmethod
    def _generate_ys(self) -> Labels:
        ...

    @classmethod
    def generate(cls, n: int) -> Dataset:
        return Dataset(n)


class SimpleDataset(Dataset):
    def __init__(self, n: int):
        super().__init__(n)

    @property
    def type(self) -> str:
        return "simple"

    def _generate_ys(self) -> Labels:
        return [1 if x1 < 0.5 else 0.0 for (x1, _) in self.xs]


class DiagonalDataset(Dataset):
    def __init__(self, n: int):
        super().__init__(n)

    @property
    def type(self) -> str:
        return "diagonal"

    def _generate_ys(self) -> Labels:
        return [1 if (x1 + x2 < 1.0) else 0.0 for (x1, x2) in self.xs]


class SplitDataset(Dataset):
    def __init__(self, n: int):
        super().__init__(n)

    @property
    def type(self) -> str:
        return "split"

    def _generate_ys(self) -> Labels:
        return [1 if (x1 < 0.2) or (x1 > 0.8) else 0.0 for (x1, _) in self.xs]


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
