from typing import Callable, List
from minitorch.autodiff.scalar import Scalar


# TODO: refactor this to actually use operators where possible!
def add(x: Scalar, y: Scalar) -> Scalar:
    """ f(x, y) = x + y """
    return x + y


def mul(x: Scalar, y: Scalar) -> Scalar:
    """ f(x, y) = x * y"""
    return x * y


def reduce(func: Callable[[Scalar, Scalar], Scalar], x0: Scalar) -> Callable[[List[Scalar]], Scalar]:
    """
    Returns a callable that applies func to each element in a list.
    """

    def reduction(ls: List[Scalar]) -> Scalar:
        # Changing list in place later on so need to make a copy
        ls_copy = ls.copy()
        x = x0
        while ls_copy:
            x = func(x, ls_copy[0])
            ls_copy.pop(0)
        return x

    return reduction


def summation(ls: List[Scalar]) -> Scalar:
    """ Sums all scalars in a list. """
    return reduce(add, x0=Scalar(0))(ls)


def product(ls: List[Scalar]) -> Scalar:
    """ Multiplies all scalars in a list. """
    return reduce(mul, x0=Scalar(1.0))(ls)
