from typing import Callable

from minitorch.constants import EPS
from minitorch.autodiff.scalar import Scalar
from minitorch.operators import is_close


def central_difference(
    func: Callable[..., Scalar], *values, arg_idx: int = 0, epsilon=1e-04
) -> Scalar:
    """
    Computes a numerical approximation of the derivative of f with respect to one arg.

    Args:
        func - Callable[..., Any]
            The function to differentiate.
        *values - List[...]
            The parameters to pass to func.
        arg_idx - int, default = 0
            The index of the variable in *values to compute the derivative with respect to.
        epsilon - float, default = 1e-06
            A small constant.
    """
    upper_values = [
        (val + epsilon) if i == arg_idx else val for (i, val) in enumerate(values)
    ]
    lower_values = [
        (val - epsilon) if i == arg_idx else val for (i, val) in enumerate(values)
    ]

    return (func(*upper_values) - func(*lower_values)) / (2 * epsilon)


def derivative_check(func: Callable[..., Scalar], *scalars):
    """
    Checks that autodiff works on an arbitrary python function.
    Asserts False if derivative is incorrect.
    """
    for scalar in scalars:
        scalar.requires_grad_(True)
    out_ = func(*scalars)
    out_.backward()

    # Run derivative check using central_difference
    for (i, scalar) in enumerate(scalars):
        check = central_difference(func, *scalars, arg_idx=i)
        print(f"now testing scalar {scalar}, {scalar.derivative}")
        if not is_close(scalar.derivative, check.data):
            raise ValueError(
                f"Derivative check failed for function {func.__name__} with arguments {scalars}. "
                f"Derivative failed at position {i}. Calculated derivative is {scalar.derivative},"
                f" should be {check.data}."
            )
