from minitorch import operators
from minitorch.central_diff import central_difference


def test_central_difference():
    d = central_difference(operators.identity, 5., arg_idx=0)
    assert operators.is_close(d, 1.)

    d = central_difference(operators.add, 5., 10., arg_idx=0)
    assert operators.is_close(d, 1.)

    d = central_difference(operators.mul, 5., 10., arg_idx=0)
    assert operators.is_close(d, 10.)

    d = central_difference(operators.exp, 2, arg_idx=0)
    assert operators.is_close(d, operators.exp(2.))
