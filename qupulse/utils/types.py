import typing
import abc
import inspect
import numbers
import fractions
import collections
import itertools
from collections.abc import Mapping as ABCMapping
import warnings

import numpy

__all__ = ["MeasurementWindow", "ChannelID", "HashableNumpyArray", "TimeType", "time_from_float", "DocStringABCMeta",
           "SingletonABCMeta"]

MeasurementWindow = typing.Tuple[str, numbers.Real, numbers.Real]
ChannelID = typing.Union[str, int]


try:
    import gmpy2
    TimeType = gmpy2.mpq

    def time_from_float(time: float, absolute_error: float=1e-12) -> TimeType:
        # gmpy2 is at least an order of magnitude faster than fractions.Fraction
        return gmpy2.mpq(gmpy2.f2q(time, absolute_error))

    def time_from_fraction(numerator: int, denominator: int = 1) -> TimeType:
        return gmpy2.mpq(numerator, denominator)

except ImportError:
    warnings.warn('gmpy2 not found. Using fractions.Fraction as fallback. Install gmpy2 for better performance.')

    TimeType = fractions.Fraction

    def time_from_float(time: float, absolute_error: float = 1e-12) -> TimeType:
        return fractions.Fraction(time).limit_denominator(int(1/absolute_error))

    def time_from_fraction(numerator: int, denominator: int = 1) -> TimeType:
        return fractions.Fraction(numerator=numerator, denominator=denominator)


class DocStringABCMeta(abc.ABCMeta):
    """Metaclass that copies/refers to docstrings of the super class."""
    def __new__(mcls, classname, bases, cls_dict):
        cls = super().__new__(mcls, classname, bases, cls_dict)

        abstract_bases = tuple(base
                               for base in reversed(inspect.getmro(cls))
                               if hasattr(base, '__abstractmethods__'))[:-1]

        for name, member in cls_dict.items():
            if not getattr(member, '__doc__'):
                if isinstance(member, property):
                    member_type = ':py:attr:'
                else:
                    member_type = ':func:'

                for base in abstract_bases:
                    if name in base.__dict__ and name in base.__abstractmethods__:
                        base_member = getattr(base, name)

                        if member is base_member or not base_member.__doc__:
                            continue

                        base_member_name = '.'.join([base.__module__, base.__qualname__, name])
                        member.__doc__ = 'Implements {}`~{}`.'.format(member_type, base_member_name)
                        break
        return cls


T = typing.TypeVar('T')


class SingletonABCMeta(DocStringABCMeta):
    """Metaclass that enforces singletons"""
    def __call__(cls: typing.Type[T]) -> T:
        return cls._instance

    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        cls._instance = super(SingletonABCMeta, cls).__call__()


class HashableNumpyArray(numpy.ndarray):
    """Make numpy arrays hashable.

    Example usage:
    my_array = np.zeros([1, 2, 3, 4])
    hashable = my_array.view(HashableNumpyArray)
    """
    def __hash__(self):
        return hash(self.tobytes())


def has_type_interface(obj: typing.Any, type_obj: typing.Type) -> bool:
    """Return true if all public attributes of the class are attribues of the object"""
    return set(dir(obj)) >= {attr for attr in dir(type_obj) if not attr.startswith('_')}


if hasattr(typing, 'Collection'):
    Collection = typing.Collection
else:
    def _check_methods(C, *methods):
        """copied from https://github.com/python/cpython/blob/3.8/Lib/_collections_abc.py"""
        mro = C.__mro__
        for method in methods:
            for B in mro:
                if method in B.__dict__:
                    if B.__dict__[method] is None:
                        return NotImplemented
                    break
            else:
                return NotImplemented
        return True

    class _ABCCollection(collections.abc.Sized, collections.abc.Iterable, collections.abc.Container):
        """copied from https://github.com/python/cpython/blob/3.8/Lib/_collections_abc.py"""
        __slots__ = ()

        @classmethod
        def __subclasshook__(cls, C):
            if cls is _ABCCollection:
                return _check_methods(C, "__len__", "__iter__", "__contains__")
            return NotImplemented

    class Collection(typing.Sized, typing.Iterable[typing.T_co], typing.Container[typing.T_co],
                     extra=_ABCCollection):
        """Fallback for typing.Collection if python 3.5
        copied from https://github.com/python/cpython/blob/3.5/Lib/typing.py"""
        __slots__ = ()
