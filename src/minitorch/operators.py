"""
Collection of core mathematical operators used through the code base.
"""

import math
from typing import Optional

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


def ge(x: float, y: float) -> float:
    """f(x, y) = 1. if x >= y else 0."""
    return 1.0 if x >= y else 0.0


def le(x: float, y: float) -> float:
    """f(x, y) = 1. if x <= y else 0."""
    return 1.0 if x <= y else 0.0


def maximum(x: float, y: float) -> float:
    """f(x, y) = x if x > y else y"""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """f(x, y) = |x - y| < 1e-03"""
    return abs(x - y) < 1e-03


def sigmoid(x: float) -> float:
    """
    f(x) = 1./(1. + e^(-x))
    Implemented as f(x) = 1./(1.+e^(-x)) if x >= 0 else e^(x)/(1. + e^(x))
    """
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """
    f(x) = x if x > 0. else 0.
    """
    return x if x > 0.0 else 0.0


def log(x: float, base: Optional[float] = None) -> float:
    """f(x) = log(x)"""
    if (base is not None) and (base > 1.0):
        return math.log(x, base)
    return math.log(x)


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
    return -d / (x**2 + EPS)


def relu_diff(x: float, d: float) -> float:
    """d * f'(x) where f(x) = relu(x)"""
    return d if x > 0 else 0.0


def sigmoid_diff(x: float, d: float) -> float:
    """d * f'(x) where f(x) = sigmoid(x) and f'(x) = sigmoid(x) * (1.0 - sigmoid(x))"""
    if x >= 0.0:
        sigmoid_x = 1.0 / (1.0 + math.exp(-x))
    else:
        sigmoid_x = math.exp(x) / (1.0 + math.exp(x))

    return d * sigmoid_x * (1.0 - sigmoid_x)
