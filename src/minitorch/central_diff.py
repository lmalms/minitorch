from typing import Callable, Any


def central_difference(func: Callable[..., Any], *values, arg_idx: int = 0, epsilon=1e-06) -> float:
    """
    Computes a numerical approximation of the derivative of f with respect to one arg.

    Args:
        func - Callable[..., Any]
            The function to differentiate.
        *values - List[...]
            The parameters to pass to func.
        arg_idx - int, default = 0
            The index of the variable in *values to compute the derivate with respect to.
        epsilon - float, default = 1e-06
            A small constant.
    """
    upper_values = [
        val
        if i != arg_idx
        else val + epsilon
        for i, val in enumerate(values)
    ]
    lower_values = [
        val
        if i != arg_idx
        else val - epsilon
        for i, val in enumerate(values)
    ]

    return (func(*upper_values) - func(*lower_values)) / epsilon
