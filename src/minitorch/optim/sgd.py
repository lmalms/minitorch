from typing import List

from minitorch.optim.base import BaseOptimizer
from minitorch.module import Parameter


class SGDOptimizer(BaseOptimizer):
    def __init__(self, parameters: List[Parameter], lr: float):
        super().__init__(parameters)
        self.lr = lr

    def step(self):
        for p in self.parameters:
            p.update(p.value - self.lr * p.value.derivative)
