from abc import ABC, abstractmethod
from typing import List

from minitorch.parameter import Parameter


class BaseOptimizer(ABC):
    def __init__(self, parameters: List[Parameter]):
        self.parameters = parameters

    def zero_grad(self) -> None:
        for p in self.parameters:
            p.zero_grad()

    @abstractmethod
    def step(self):
        ...