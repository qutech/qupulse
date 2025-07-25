# SPDX-FileCopyrightText: 2014-2024 Quantum Technology Group and Chair of Software Engineering, RWTH Aachen University
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""This package contains utility functions and classes as well as custom sympy extensions(hacks)."""

from typing import Union, Iterable, Any, Tuple, Mapping, Iterator, TypeVar, Sequence, AbstractSet, Optional, Callable
import itertools
import re
import numbers
from collections import OrderedDict
from frozendict import frozendict
from qupulse.expressions import ExpressionScalar, ExpressionLike

import numpy

try:
    from math import isclose
except ImportError:
    # py version < 3.5
    isclose = None

try:
    from functools import cached_property
except ImportError:
    # py version < 3.8
    from cached_property import cached_property

_T = TypeVar('_T')


__all__ = ["checked_int_cast", "is_integer", "isclose", "pairwise", "replace_multiple", "cached_property",
           "forced_hash", "to_next_multiple"]


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


def _fallback_pairwise(iterable: Iterable[_T]) -> Iterator[Tuple[_T, _T]]:
    """s -> (s0,s1), (s1,s2), (s2, s3), ...

    Args:
        iterable: Iterable to iterate over pairwise

    Returns:
        An iterable that yields neighbouring elements
    """
    return zip(iterable, itertools.islice(iterable, 1, None))


if hasattr(itertools, 'pairwise'):
    pairwise = itertools.pairwise
else:
    # py version < 3.10
    pairwise = _fallback_pairwise


def grouper(iterable: Iterable[Any], n: int, fillvalue=None) -> Iterable[Tuple[Any, ...]]:
    """Collect data into fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    # this is here instead of using more_itertools because there were problems with the old version's argument order
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)


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


def forced_hash(obj) -> int:
    """Try to produce a hash from obj by nested conversions to hashable types.

    Mapping -> frozendict
    AbstractSet -> frozenset
    ndarray -> bytes or nested tuples
    Sequence -> tuple
    """
    try:
        return hash(obj)
    except TypeError:
        if isinstance(obj, Mapping):
            return hash(frozendict((key, forced_hash(value))
                                   for key, value in obj.items()))
        if isinstance(obj, AbstractSet):
            return hash(frozenset(map(forced_hash, obj)))

        if isinstance(obj, numpy.ndarray):
            # case where dtype maybe has a custom hash that is not binary content dependent
            if obj.dtype == numpy.dtype('O') or not obj.dtype.isbuiltin != 1:
                return forced_hash(obj.tolist())
            else:
                return hash((obj.tobytes(), obj.shape, obj.dtype))

        if isinstance(obj, Sequence):
            return hash(tuple(map(forced_hash, obj)))

        raise


def to_next_multiple(sample_rate: ExpressionLike, quantum: int,
                     min_quanta: Optional[int] = None) -> Callable[[ExpressionLike],ExpressionScalar]:
    """Construct a helper function to expand a duration to one corresponding to
    valid sample multiples according to the arguments given.
    Useful e.g. for PulseTemplate.pad_to's 'to_new_duration'-argument.

    Args:
        sample_rate: sample rate with respect to which the duration is evaluated.
        quantum: number of samples to whose next integer multiple the duration shall be rounded up to.
        min_quanta: number of multiples of quantum not to fall short of.
    Returns:
        A function that takes a duration (ExpressionLike) as input, and returns
        a duration rounded up to the next valid samples count in given sample rate.
        The function returns 0 if duration==0, <0 is not checked if min_quanta is None.

    """
    sample_rate = ExpressionScalar(sample_rate)
    #is it more efficient to omit the Max call if not necessary?
    if min_quanta is None:
        #double negative for ceil division.
        return lambda duration: -(-(duration*sample_rate)//quantum) * (quantum/sample_rate)
    else:
        #still return 0 if duration==0
        return lambda duration: ExpressionScalar(f'{quantum}/({sample_rate})*Max({min_quanta},-(-{duration}*{sample_rate}//{quantum}))*Max(0, sign({duration}))')
   