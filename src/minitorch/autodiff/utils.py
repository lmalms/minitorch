from typing import Union, Tuple, Any


def wrap_tuple(x: Any) -> Tuple[Any]:
    if isinstance(x, tuple):
        return x
    return (x, )


def unwrap_tuple(x: Union[Tuple[Any], Any]) -> Any:
    if len(x) == 1:
        return x[0]
    return x
