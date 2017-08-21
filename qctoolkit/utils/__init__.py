from typing import Union

import numpy

__all__ = ["checked_int_cast"]


def checked_int_cast(x: Union[float, int, numpy.ndarray], epsilon: float=1e-10) -> int:
    if isinstance(x, numpy.ndarray):
        if len(x) != 1:
            raise ValueError('Not a scalar value')
    if isinstance(x, int):
        return x
    int_x = int(round(x))
    if abs(x - int_x) > epsilon:
        raise ValueError('No integer', x)
    return int_x
