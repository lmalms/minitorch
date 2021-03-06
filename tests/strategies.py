from hypothesis import settings
from hypothesis.strategies import floats, integers

from minitorch.operators import is_close

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

small_ints = integers(min_value=1, max_value=3)
small_floats = floats(min_value=-100, max_value=100, allow_nan=False)
tiny_floats = floats(min_value=-5, max_value=5, allow_nan=False)
small_positive_floats = floats(min_value=0.1, max_value=100, allow_nan=False)
tiny_positive_floats = floats(min_value=0.1, max_value=10, allow_nan=False)
med_ints = integers(min_value=1, max_value=20)


def assert_close(a: float, b: float) -> None:
    assert is_close(a, b), f"failure with a={a} and b={b}."
