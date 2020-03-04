from typing import Union, Iterable, Any, Tuple, Mapping
import itertools
import re
import numbers
from collections import OrderedDict

import numpy

try:
    from math import isclose
except ImportError:
    # py version < 3.5
    isclose = None

__all__ = ["checked_int_cast", "is_integer", "isclose", "pairwise", "replace_multiple"]


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


def is_integer(x: numbers.Real, epsilon: float=1e-6) -> bool:
    return abs(x - int(round(x))) < epsilon


def _fallback_is_close(a, b, *, rel_tol=1e-09, abs_tol=0.0):
    """Copied from https://docs.python.org/3/library/math.html
    Does no error checks."""
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)  # pragma: no cover


if not isclose:
    isclose = _fallback_is_close


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


def replace_multiple(s: str, replacements: Mapping[str, str]) -> str:
    """Replace multiple strings at once. If multiple replacements overlap the precedence is given by the order in
    replacements.

    For pyver >= 3.6 (otherwise use OrderedDict)
    >>> assert replace_multiple('asdf', {'asd': '1', 'asdf', '2'}) == 'asd1'
    >>> assert replace_multiple('asdf', {'asdf': '2', 'asd', '1'}) == '2'
    """
    rep = OrderedDict((re.escape(k), v) for k, v in replacements.items())
    pattern = re.compile("|".join(rep.keys()))
    return pattern.sub(lambda m: rep[re.escape(m.group(0))], s)
