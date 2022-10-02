from typing import Optional

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

from minitorch.autodiff import Index, Shape, TensorData
from minitorch.functional import product

from .strategies import small_ints

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


@composite
def vals(draw_fn, size: int, number: SearchStrategy[float]):  # returns Tensor
    pass


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
