import random
from abc import abstractmethod
from typing import List, Tuple

random.seed(0)


class Dataset:
    def __init__(self, n: int):
        self.n = n
        self._xs = self._generate_xs()
        self._ys = self._generate_ys()

    @property
    def xs(self) -> List[Tuple[float, float]]:
        return self._xs

    @property
    def ys(self) -> List[float]:
        return self._ys

    def _generate_xs(self) -> List[Tuple[float, float]]:
        return [(random.random(), random.random()) for _ in range(self.n)]

    @abstractmethod
    def _generate_ys(self) -> List[float]:
        ...


class SimpleDataset(Dataset):
    def __init__(self, n: int):
        super().__init__(n)

    def _generate_ys(self) -> List[float]:
        return [1 if x1 < 0.5 else 0.0 for (x1, _) in self.xs]


class DiagonalDataset(Dataset):
    def __init__(self, n: int):
        super().__init__(n)

    def _generate_ys(self) -> List[float]:
        return [1 if (x1 + x2 < 1.0) else 0.0 for (x1, x2) in self.xs]


class SplitDataset(Dataset):
    def __init__(self, n: int):
        super().__init__(n)

    def _generate_ys(self) -> List[float]:
        return [1 if (x1 < 0.2) or (x1 > 0.8) else 0.0 for (x1, _) in self.xs]


class XORDataset(Dataset):
    def __init__(self, n: int):
        super().__init__(n)

    def _generate_ys(self) -> List[float]:
        return [
            1 if ((x1 < 0.5) and (x2 > 0.5)) or ((x1 > 0.5) and (x2 < 0.5)) else 0.0
            for (x1, x2) in self.xs
        ]
