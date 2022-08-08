from typing import Any, Tuple, Union


def wrap_tuple(x: Any) -> Tuple[Any]:
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x: Union[Tuple[Any], Any]) -> Any:
    if (isinstance(x, tuple)) and (len(x) == 1):
        return x[0]
    return x
