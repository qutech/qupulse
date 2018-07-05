from typing import Union, Iterable, Any, Tuple
import itertools

import numpy

__all__ = ["checked_int_cast", "is_integer", "pairwise"]


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


def pairwise(iterable: Iterable[Any],
             zip_function=itertools.zip_longest, **kwargs) -> Iterable[Tuple[Any, Any]]:
    """s -> (s0,s1), (s1,s2), (s2, s3), ...

    Args:
        iterable: Iterable to iterate over pairwise
        zip_function: Either zip or itertools.zip_longest(default)
        **kwargs: Gets passed to zip_function

    Returns:
        An iterable that yield neighbouring elements
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip_function(a, b, **kwargs)
