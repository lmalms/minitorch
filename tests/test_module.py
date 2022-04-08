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
