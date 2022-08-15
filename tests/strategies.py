from hypothesis import settings
from hypothesis.strategies import composite, floats, integers

from minitorch.autodiff import Scalar
from minitorch.functional import operators

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


def assert_close(a: float, b: float) -> None:
    assert operators.is_close(a, b), f"failure with a={a} and b={b}."


@composite
def scalars(draw, min_value=-1e06, max_value=1e06):
    value = draw(floats(min_value=min_value, max_value=max_value, allow_nan=False))
    return Scalar(value)


small_ints = integers(min_value=1, max_value=3)
small_floats = floats(min_value=-100, max_value=100, allow_nan=False)
tiny_floats = floats(min_value=-5, max_value=5, allow_nan=False)
small_positive_floats = floats(min_value=0.1, max_value=100, allow_nan=False)
tiny_positive_floats = floats(min_value=0.1, max_value=10, allow_nan=False)
med_ints = integers(min_value=1, max_value=20)


small_scalars = scalars(min_value=-100, max_value=100)
