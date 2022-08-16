from minitorch.autodiff import central_difference
from minitorch.functional import operators


def test_central_difference():
    d = central_difference(operators.identity, 5.0, arg_idx=0)
    assert operators.is_close(d, 1.0)

    d = central_difference(operators.add, 5.0, 10.0, arg_idx=0)
    assert operators.is_close(d, 1.0)

    d = central_difference(operators.mul, 5.0, 10.0, arg_idx=0)
    assert operators.is_close(d, 10.0)

    d = central_difference(operators.exp, 2, arg_idx=0)
    assert operators.is_close(d, operators.exp(2.0))
