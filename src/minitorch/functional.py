from typing import Callable, List, Union

from typing_extensions import TypeAlias

from minitorch.operators import add, mul, neg

ScalarLike: TypeAlias = Union[float, int, "Scalar"]


def reduce(
    func: Callable[[ScalarLike, ScalarLike], ScalarLike], x0: ScalarLike
) -> Callable[[List[ScalarLike]], ScalarLike]:
    """
    Returns a callable that applies func to each element in a list.
    """

    def reduction(ls: List[ScalarLike]) -> ScalarLike:
        # Changing list in place later on so need to make a copy
        ls_copy = ls.copy()
        x = x0
        while ls_copy:
            x = func(x, ls_copy[0])
            ls_copy.pop(0)
        return x

    return reduction


def map_single(
    func: Callable[[ScalarLike], ScalarLike]
) -> Callable[[List[ScalarLike]], List[ScalarLike]]:
    """
    Higher order map. Returns mapping function that applies func to
    every element in a list and then returns that list.
    """

    def mapping_func(ls: List[ScalarLike]) -> List[ScalarLike]:
        return [func(i) for i in ls]

    return mapping_func


def map_double(
    func: Callable[[ScalarLike, ScalarLike], ScalarLike]
) -> Callable[[List[ScalarLike], List[ScalarLike]], List[ScalarLike]]:
    """
    Higher order map. Returns a mapping function that applies func to elements in two equally sized list and returns
    result as new list.
    """

    def mapping_func(ls1: List[ScalarLike], ls2: List[ScalarLike]) -> List[ScalarLike]:
        return [func(i, j) for (i, j) in zip(ls1, ls2)]

    return mapping_func


def summation(ls: List[ScalarLike]) -> ScalarLike:
    """Sums all values in a list."""
    return reduce(add, x0=0.0)(ls)


def product(ls: List[ScalarLike]) -> ScalarLike:
    """Multiplies all scalars in a list."""
    return reduce(mul, x0=1.0)(ls)


def neg_list(ls: List[ScalarLike]) -> List[ScalarLike]:
    """
    Negates each element in ls.
    """
    return map_single(neg)(ls)


def add_lists(ls1: List[ScalarLike], ls2: List[ScalarLike]) -> List[ScalarLike]:
    """
    Sums elements in lists ls1 and ls2 pairwise and returns result as new list.
    """
    return map_double(add)(ls1, ls2)


def multiply_lists(ls1: List[ScalarLike], ls2: List[ScalarLike]) -> List[ScalarLike]:
    """
    Pairwise multiplication of elements in two lists.
    """
    return map_double(mul)(ls1, ls2)
