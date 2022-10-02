"""
Implementations of the autodifferentiation Functions for Tensors.
"""

from __future__ import annotations

import random
from abc import abstractmethod
from typing import Iterable, Tuple

from minitorch.autodiff import Context, History
from minitorch.autodiff.utils import wrap_tuple


class BaseTensorFunction:

    # TODO: check that the implementation here matches ScalarBaseFunction
    #  can I actually inherit from Function and write tensor functions?

    @classmethod
    @abstractmethod
    def backward(cls, ctx: Context, out_: Tensor) -> Tuple[Tensor, ...]:
        ...

    @classmethod
    @abstractmethod
    def forward(cls, ctx: Context, *inputs: Iterable[Tensor]):
        ...

    @classmethod
    def _backward(cls, ctx: Context, out_: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, out_))

    @classmethod
    def _forward(cls, ctx: Context, *inputs: Iterable[Tensor]) -> Tensor:
        return cls.forward(ctx, *inputs)

    @classmethod
    def apply(cls, *tensors: Iterable[Tensor]) -> Tensor:
        raw_values = []
        requires_grad = False
        for t in tensors:
            if t.history is not None:
                requires_grad = True
            raw_values.append(t.detach())

        # Create new context for Tensor
        ctx = Context(requires_grad)

        # Call forward with input values
        c = cls._forward(ctx, *raw_values)

        back = History(last_fn=cls, ctx=ctx, inputs=tensors) if requires_grad else None
        return Tensor(c._tensor, back, backend=c.backend)
