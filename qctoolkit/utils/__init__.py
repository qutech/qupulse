from typing import Union

__all__ = ["checked_int_cast"]


def checked_int_cast(x: Union[float, int], epsilon: float=1e-10) -> int:
    if isinstance(x, int):
        return x
    int_x = int(round(x))
    if abs(x - int_x) > epsilon:
        raise ValueError('No integer', x)
    return int_x
