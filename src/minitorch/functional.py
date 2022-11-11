from typing import Callable, List, Union

from typing_extensions import TypeAlias

import minitorch.autodiff.variable as variable
from minitorch.operators import add, mul, neg

VariableLike: TypeAlias = Union[int, float, variable.Variable]


def reduce(
    fn: Callable[[VariableLike, VariableLike], VariableLike],
    x0: VariableLike,
) -> Callable[[List[VariableLike]], VariableLike]:
    """
    Returns a callable that applies func to each element in a list.
    """

    def reduction(ls: List[VariableLike]) -> VariableLike:
        # Changing list in place later on so need to make a copy
        ls_copy = ls.copy()
        x = x0
        while ls_copy:
            x = fn(x, ls_copy[0])
            ls_copy.pop(0)
        return x

    return reduction


def map_single(
    fn: Callable[[VariableLike], VariableLike]
) -> Callable[[List[VariableLike]], List[VariableLike]]:
    """
    Higher order map. Returns mapping function that applies func to
    every element in a list and then returns that list.
    """

    def mapping_func(ls: List[VariableLike]) -> List[VariableLike]:
        return [fn(i) for i in ls]

    return mapping_func


def map_double(
    fn: Callable[[VariableLike, VariableLike], VariableLike]
) -> Callable[[List[VariableLike], List[VariableLike]], List[VariableLike]]:
    """
    Higher order map. Returns a mapping function that applies func to elements in two equally sized list and returns
    result as new list.
    """

    def mapping_func(
        ls1: List[VariableLike], ls2: List[VariableLike]
    ) -> List[VariableLike]:
        return [fn(i, j) for (i, j) in zip(ls1, ls2)]

    return mapping_func


def summation(ls: List[VariableLike]) -> VariableLike:
    """Sums all values in a list."""
    return reduce(add, x0=0.0)(ls)


def product(ls: List[VariableLike]) -> VariableLike:
    """Multiplies all scalars in a list."""
    return reduce(mul, x0=1.0)(ls)


def neg_list(ls: List[VariableLike]) -> List[VariableLike]:
    """
    Negates each element in ls.
    """
    return map_single(neg)(ls)


def add_lists(ls1: List[VariableLike], ls2: List[VariableLike]) -> List[VariableLike]:
    """
    Sums elements in lists ls1 and ls2 pairwise and returns result as new list.
    """
    return map_double(add)(ls1, ls2)


def multiply_lists(
    ls1: List[VariableLike], ls2: List[VariableLike]
) -> List[VariableLike]:
    """
    Pairwise multiplication of elements in two lists.
    """
    return map_double(mul)(ls1, ls2)
