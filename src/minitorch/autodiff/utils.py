from typing import Any, Tuple, TypeVar, Union

T = TypeVar("T")


def wrap_tuple(x: T) -> Tuple[T]:
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x: Union[Tuple[T], T]) -> T:
    if (isinstance(x, tuple)) and (len(x) == 1):
        return x[0]
    return x
