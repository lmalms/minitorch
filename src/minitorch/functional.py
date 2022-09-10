from typing import Callable, List

from minitorch.operators import add, mul, neg
from minitorch.types import FloatOrScalar


def reduce(
    func: Callable[[FloatOrScalar, FloatOrScalar], FloatOrScalar], x0: FloatOrScalar
) -> Callable[[List[FloatOrScalar]], FloatOrScalar]:
    """
    Returns a callable that applies func to each element in a list.
    """

    def reduction(ls: List[FloatOrScalar]) -> FloatOrScalar:
        # Changing list in place later on so need to make a copy
        ls_copy = ls.copy()
        x = x0
        while ls_copy:
            x = func(x, ls_copy[0])
            ls_copy.pop(0)
        return x

    return reduction


def summation(ls: List[FloatOrScalar]) -> FloatOrScalar:
    """Sums all values in a list."""
    return reduce(add, x0=0.0)(ls)


def product(ls: List[FloatOrScalar]) -> FloatOrScalar:
    """Multiplies all scalars in a list."""
    return reduce(mul, x0=1.0)(ls)


def map_single(
    func: Callable[[FloatOrScalar], FloatOrScalar]
) -> Callable[[List[FloatOrScalar]], List[FloatOrScalar]]:
    """
    Higher order map. Returns mapping function that applies func to
    every element in a list and then returns that list.
    """

    def mapping_func(ls: List[FloatOrScalar]) -> List[FloatOrScalar]:
        return [func(i) for i in ls]

    return mapping_func


def map_double(
    func: Callable[[FloatOrScalar, FloatOrScalar], FloatOrScalar]
) -> Callable[[List[FloatOrScalar], List[FloatOrScalar]], List[FloatOrScalar]]:
    """
    Higher order map. Returns a mapping function that applies func to elements in two equally sized list and returns
    result as new list.
    """

    def mapping_func(
        ls1: List[FloatOrScalar], ls2: List[FloatOrScalar]
    ) -> List[FloatOrScalar]:
        return [func(i, j) for (i, j) in zip(ls1, ls2)]

    return mapping_func


def neg_list(ls: List[FloatOrScalar]) -> List[FloatOrScalar]:
    """
    Negates each element in ls.
    """
    return map_single(neg)(ls)


def add_lists(
    ls1: List[FloatOrScalar], ls2: List[FloatOrScalar]
) -> List[FloatOrScalar]:
    """
    Sums elements in lists ls1 and ls2 pairwise and returns result as new list.
    """
    return map_double(add)(ls1, ls2)
