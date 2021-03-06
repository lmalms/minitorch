"""
Collection of core mathematical operators used through the code base.
"""

import math
from typing import Callable, List, Optional

from minitorch.constants import EPS


def mul(x: float, y: float) -> float:
    """f(x, y) = x * y"""
    return x * y


def identity(x: float) -> float:
    """Identity function. f(x) = x"""
    return x


def add(x: float, y: float) -> float:
    """f(x, y) = x + y"""
    return x + y


def neg(x: float) -> float:
    """f(x) = -x"""
    return -x


def lt(x: float, y: float) -> float:
    """f(x, y) = 1. if x < y else 0."""
    return 1.0 if x < y else 0.0


def gt(x: float, y: float) -> float:
    """f(x, y) = 1. if x > y else 0."""
    return 1.0 if x > y else 0.0


def eq(x: float, y: float) -> float:
    """f(x, y) = 1. if x == y else 0."""
    return 1.0 if x == y else 0.0


def maximum(x: float, y: float) -> float:
    """f(x, y) = x if x > y else y"""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """f(x, y) = |x - y| < 1e-02"""
    return abs(x - y) < 1e-03


def sigmoid(x: float) -> float:
    """
    f(x) = 1./(1. + e^(-x))
    Implemented as f(x) = 1./(1.+e^(-x)) if x >= 0 else e^(x)/(1. + e^(x))
    """
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0.0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """
    f(x) = x if x > 0. else 0.
    """
    return x if x > 0.0 else 0.0


def log(x: float, base: Optional[float] = None) -> float:
    """f(x) = log(x)"""
    return math.log(x, base) if base is not None else math.log(x)


def exp(x: float) -> float:
    """f(x) = e^(x)"""
    return math.exp(x)


def log_diff(x: float, d: float) -> float:
    """d * f'(x) where f(x) = log(x)"""
    return d / (x + EPS)


def inv(x: float) -> float:
    """f(x) = 1 / x"""
    return 1.0 / (x + EPS)


def inv_diff(x: float, d: float) -> float:
    """d * f'(x) where f(x) = 1/x"""
    return d / (x**2)


def relu_diff(x: float, d: float) -> float:
    """d * f'(x) where f(x) = relu(x)"""
    return d if x > 0 else 0.0


def sigmoid_diff(x: float, d: float) -> float:
    """d * f'(x) where f(x) = sigmoid(x)"""
    return d * sigmoid(x) * (1.0 - sigmoid(x))


def map_single(func: Callable[[float], float]) -> Callable[[List[float]], List[float]]:
    """
    Higher order map. Returns mapping function that applies func to every element in a list and then returns that list.
    """

    def mapping_func(ls: List[float]) -> List[float]:
        return [func(i) for i in ls]

    return mapping_func


def map_double(
    func: Callable[[float, float], float]
) -> Callable[[List[float], List[float]], List[float]]:
    """
    Higher order map. Returns a mapping function that applies func to elements in two equally sized list and returns
    result as new list.
    """

    def mapping_func(ls1: List[float], ls2: List[float]) -> List[float]:
        return [func(i, j) for (i, j) in zip(ls1, ls2)]

    return mapping_func


def neg_list(ls: List[float]) -> List[float]:
    """
    Negates each element in ls.
    """
    return map_single(neg)(ls)


def add_lists(ls1: List[float], ls2: List[float]) -> List[float]:
    """
    Sums elements in lists ls1 and ls2 pairwise and returns result as new list.
    """
    return map_double(add)(ls1, ls2)


def reduce(
    func: Callable[[float, float], float], x0: float
) -> Callable[[List[float]], float]:
    """
    Returns function that applies reduction to each element in list.
    """

    def reduction(ls: List[float]) -> float:
        ls_copy = ls.copy()  # changing to original list so make a copy.
        x = x0
        while ls_copy:
            x = func(x, ls_copy[0])
            ls_copy.pop(0)
        return x

    return reduction


def summation(ls: List[float]) -> float:
    """
    Computes the sum for all elements in a list.
    """
    return reduce(add, x0=0.0)(ls)


def product(ls: List[float]) -> float:
    """
    Computes the product of all elements in a list.
    """
    return reduce(mul, x0=1.0)(ls)
