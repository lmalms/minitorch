from typing import List

from minitorch.module.parameter import Parameter
from minitorch.optim.base import BaseOptimizer


class SGDOptimizer(BaseOptimizer):
    def __init__(self, parameters: List[Parameter], lr: float):
        super().__init__(parameters)
        self.lr = lr

    def step(self) -> None:
        for p in self.parameters:
            p.update(p.value - self.lr * p.value.derivative)
