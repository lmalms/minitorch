from typing import Union, Tuple, Any


def wrap_tuple(x: Any) -> Tuple[Any]:
    # TODO: validate types here.
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x: Union[Tuple[float], float]) -> float:
    # TODO: have some type validation here.
    # TODO: check are these really floats? or Variables?
    if (isinstance(x, tuple)) and (len(x) == 1):
        return x[0]
    return x


def derivative_check():
    pass
