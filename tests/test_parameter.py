import pytest

from src.minitorch.parameter import Parameter


class MockParam:
    def __init__(self):
        self._requires_grad_: bool = False

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad_

    def requires_grad_(self, requires_grad: bool):
        self._requires_grad_ = requires_grad


def test_parameter():
    param1 = MockParam()
    param2 = Parameter(value=param1)
    assert param1.requires_grad
    print(param2)


def test_update():
    param1 = MockParam()
    param2 = MockParam()
    param3 = Parameter(value=param1)
    param3.update(value=param2)
    assert param2.requires_grad
