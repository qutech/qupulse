from typing import Union

import numpy

try:
    from math import isclose
except ImportError:
    # py version < 3.5
    isclose = None

__all__ = ["checked_int_cast", "is_integer", "isclose"]


def checked_int_cast(x: Union[float, int, numpy.ndarray], epsilon: float=1e-6) -> int:
    if isinstance(x, numpy.ndarray):
        if len(x) != 1:
            raise ValueError('Not a scalar value')
    if isinstance(x, int):
        return x
    int_x = int(round(x))
    if abs(x - int_x) > epsilon:
        raise ValueError('No integer', x)
    return int_x


def is_integer(x: Union[float, int], epsilon: float=1e-6) -> bool:
    return abs(x - int(round(x))) < epsilon


def _fallback_is_close(a, b, *, rel_tol=1e-09, abs_tol=0.0):
    """Copied from https://docs.python.org/3/library/math.html
    Does no error checks."""
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)  # pragma: no cover


if not isclose:
    isclose = _fallback_is_close
