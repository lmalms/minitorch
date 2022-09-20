from typing import List, Optional

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

from minitorch.autodiff import TensorData, UserShape
from minitorch.functional import product

from .strategies import small_ints

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


@composite
def vals(draw_fn, size: int, number: SearchStrategy[float]):  # returns Tensor
    pass


@composite
def shapes(draw: DrawFn) -> UserShape:
    shape = draw(lists(small_ints, min_size=1, max_size=4))
    return tuple(shape)


@composite
def tensor_data(
    draw: DrawFn,
    numbers: SearchStrategy[float] = floats(),
    shape: Optional[UserShape] = None,
) -> TensorData:
    if shape is None:
        shape = draw(shapes())
    size = int(product(list(shape)))
    data = draw(lists(numbers, min_size=size, max_size=size))

    permute = draw(permutations(range(len(shape))))
    permute_shape = tuple([shape[i] for i in permute])

    z = sorted(enumerate(permute), key=lambda a: a[1])
    reverse_permute = [a[0] for a in z]
    td = TensorData(data, permute_shape)
    ret = td.permute(*reverse_permute)
    assert ret.shape == shape[0]
    return ret
