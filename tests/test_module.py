import pytest

from hypothesis import given
from .strategies import med_ints, small_floats
from src.minitorch.module import Module
from src.minitorch.parameter import Parameter


class ModuleA1(Module):
    def __init__(self):
        super().__init__()
        self.p1 = Parameter(value=15., name="p1")


class ModuleA2(Module):
    def __init__(self):
        super().__init__()
        self.p2 = Parameter(value=10., name="p2")


class ModuleA3(Module):
    def __init__(self):
        super().__init__()
        self.p3 = Parameter(value=3.3, name="p3")
        self.c = ModuleA2()


class ModuleA4(Module):
    def __init__(self):
        super().__init__()
        self.p4 = Parameter(value=5., name="p4")
        self.non_param = 7
        self.a = ModuleA1()
        self.b = ModuleA3()


def test_moduleA_stack():
    """
    Check that each of the properties match.
    """
    module = ModuleA4()
    named_parameters = dict(module.named_parameters())
    print(named_parameters)
    print(module.p4)

    print(str(module))
    assert module.p4.value == 5.
    assert module.non_param == 7.
    assert module.a.p1.value == 15
    assert module.b.p3.value == 3.3
    assert module.b.c.p2.value == 10

    # Checked named parameters
    assert named_parameters["p4"].value == 5.
    assert named_parameters["a.p1"].value == 15.
    assert named_parameters["b.p3"].value == 3.3
    assert named_parameters["b.c.p2"].value == 10.


VAL_A = 50
VAL_B = 100


class ModuleB1(Module):
    def __init__(self):
        super().__init__()
        self.parameter_a = Parameter(value=VAL_A)


class ModuleB2(Module):
    def __init__(self, n_extra: int = 0):
        super().__init__()
        self.parameter_a = Parameter(VAL_A)
        self.parameter_b = Parameter(VAL_B)
        self.non_parameter = 10.
        self.module_a = ModuleB1()
        for i in range(n_extra):
            self.add_parameter(Parameter(value=None, name=f"extra_parameter_{i}"))


class ModuleB3(Module):
    def __init__(self, size_module_a: int, size_module_b: int, param_value: float):
        super().__init__()
        self.module_a = ModuleB2(n_extra=size_module_a)
        self.module_b = ModuleB2(n_extra=size_module_b)
        self.parameter_val = Parameter(param_value)


@given(med_ints, med_ints)
def test_moduleB(size_a: int, size_b: int):
    """Verify properties of single modules."""
    module = ModuleB2()
    module.eval()
    assert not module.training
    module.train()
    assert module.training
    assert len(module.parameters()) == 3

    module = ModuleB2(size_a)
    assert len(module.parameters()) == size_a + 3

    module = ModuleB2(size_b)
    named_parameters = dict(module.named_parameters())
    assert named_parameters["parameter_a"].value == VAL_A
    assert named_parameters["parameter_b"].value == VAL_B
    assert named_parameters["extra_parameter_0"].value is None
    assert named_parameters["module_a.parameter_a"].value == VAL_A


@given(med_ints, med_ints, small_floats)
def test_moduleB_stack(size_a: int, size_b: int, val: float):
    """Verify properties of a stacked module"""
    module = ModuleB3(size_a, size_b, param_value=val)
    module.eval()
    assert not module.training
    assert not module.module_a.training
    assert not module.module_b.training
    assert not module.module_a.module_a.training
    assert not module.module_b.module_a.training

    module.train()
    assert module.training
    assert module.module_a.training
    assert module.module_b.training
    assert module.module_a.module_a.training
    assert module.module_b.module_a.training

    assert len(module.parameters()) == 1 + (size_a + 3) + (size_b + 3)

    named_parameters = dict(module.named_parameters())
    assert named_parameters["parameter_val"].value == val
    assert named_parameters["module_a.parameter_a"].value == VAL_A
    assert named_parameters["module_a.parameter_b"].value == VAL_B
    assert named_parameters["module_a.module_a.parameter_a"].value == VAL_A

    assert named_parameters["module_b.parameter_a"].value == VAL_A
    assert named_parameters["module_b.parameter_b"].value == VAL_B
    assert named_parameters["module_b.module_a.parameter_a"].value == VAL_A


class ModuleC1(Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs) -> float:
        return 10.


@pytest.mark.xfail
def test_forward():
    module = Module()
    module()

    module = ModuleC1()
    assert module.forward() == 10.
    assert module() == 10.