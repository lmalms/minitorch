from typing import List, Optional

import numpy as np
from hypothesis import settings
from hypothesis.strategies import (
    DrawFn,
    SearchStrategy,
    composite,
    floats,
    integers,
    lists,
    permutations,
)

from minitorch.autodiff import Tensor, TensorData, TensorBackend, tensor, SimpleOps
from minitorch.autodiff.tensor_data import Index, Shape
from minitorch.functional import product

from .strategies import small_ints

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


DEFAULT_FLOAT_SEARCH_STRATEGY = floats(allow_nan=False, min_value=-100, max_value=100)


@composite
def vals(draw_fn: DrawFn, size: int, number: SearchStrategy[float]) -> Tensor:
    data = draw_fn(lists(number, min_size=size, max_size=size))
    return tensor(data)


@composite
def shapes(draw: DrawFn) -> Shape:
    shape = draw(lists(small_ints, min_size=1, max_size=4))
    return tuple(shape)


@composite
def indices(draw: DrawFn, layout: TensorData) -> Index:
    return tuple((draw(integers(min_value=0, max_value=s - 1)) for s in layout.shape))


@composite
def tensor_data(
    draw: DrawFn,
    numbers: SearchStrategy[float] = floats(),
    shape: Optional[Shape] = None,
) -> TensorData:
    if shape is None:
        shape = draw(shapes())
    size = int(product(list(shape)))
    data = draw(lists(numbers, min_size=size, max_size=size))

    permutation = draw(permutations(range(len(shape))))
    shape_permutation = tuple([shape[i] for i in permutation])
    reverse_permutation = [
        i for (i, _) in sorted(enumerate(permutation), key=lambda pair: pair[1])
    ]

    # Applying the reverse permutation to permutation
    # should recover original shape
    td = TensorData(data, shape_permutation)
    reverse_td = td.permute(*reverse_permutation)

    assert np.all(reverse_td.shape == np.array(shape))
    return reverse_td


@composite
def tensors(
    draw_fn: DrawFn,
    numbers: SearchStrategy[float] = DEFAULT_FLOAT_SEARCH_STRATEGY,
    backend: TensorBackend = TensorBackend(SimpleOps),
    shape: Optional[Shape] = None,
) -> Tensor:
    td = draw_fn(tensor_data(numbers, shape=shape))
    return Tensor(td, backend=backend)


@composite
def shaped_tensors(
    draw_fn: DrawFn,
    n: int,
    numbers: SearchStrategy[float] = DEFAULT_FLOAT_SEARCH_STRATEGY,
    backend: TensorBackend = TensorBackend(SimpleOps),
) -> List[Tensor]:

    td = draw_fn(tensor_data(numbers))

    tensors = []
    for i in range(n):
        data = draw_fn(lists(numbers, min_size=td.size, max_size=td.size))
        tensors.append(
            Tensor(data=TensorData(data, td.shape, td.strides), backend=backend)
        )
    return tensors
