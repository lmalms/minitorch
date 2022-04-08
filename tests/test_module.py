import pytest

from hypothesis import given
from .strategies import med_ints, small_floats
from minitorch.module import Module
from minitorch.parameter import Parameter


class ModuleA1(Module):
    def __init__(self):
        super().__init__()
        self.p1 = Parameter(value=15, name="p1")


class ModuleA2(Module):
    def __init__(self):
        super().__init__()
        self.p2 = Parameter(value=10, name="p2")


class ModuleA3(Module):
    def __init__(self):
        super().__init__()
        self.p3 = Parameter(value=3.3, name="p3")
        self.c = ModuleA2()


class ModuleA4(Module):
    def __init__(self):
        super().__init__()
        self.p4 = Parameter(value=5, name="p4")
        self.non_param = 7
        self.a = ModuleA1()
        self.b = ModuleA3()


def test_stacked_module():
    """
    Check that each of the properties match.
    """
    module = ModuleA4()
    named_parameters = dict(module.named_parameters())

    print(str(module))
    assert module.p4 == 5
    assert module.non_param == 7
    assert module.a.p1 == 15
    assert module.b.p3 == 3.3
    assert module.b.c.p2 == 10

    # Checked named parameters
    assert named_parameters["p4"] == 5
    assert named_parameters["a.p1"] == 15
    assert named_parameters["b.p3"] == 3.3
    assert named_parameters["b.c.p2"] == 10
