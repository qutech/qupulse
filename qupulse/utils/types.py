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
except ImportError:
    gmpy2 = None

    warnings.warn('gmpy2 not found. Using fractions.Fraction as fallback. Install gmpy2 for better performance.'
                  'time_from_float might produce slightly different results')


def _with_other_as_time_type(fn):
    """This is decorator to convert the other argument into a TimeType"""
    def wrapper(self, other) -> 'TimeType':
        converted = _converter.get(type(other), TimeType)(other)
        result = fn(self, converted)
        if result is NotImplemented:
            return result
        elif type(result) is TimeType._InternalRepresentation:
            return TimeType(result)
        else:
            return result
    return wrapper


class TimeType(numbers.Rational):
    """This type represents a rational number with arbitrary precision.

    Internally it uses gmpy2.mpq (if available) or fractions.Fraction
    """
    __slots__ = ('_value',)

    _InternalRepresentation = fractions.Fraction if gmpy2 is None else gmpy2.mpq

    def __init__(self, value: numbers.Real = 0.):
        if type(value) == type(self):
            self._value = value._value
        else:
            self._value = self._InternalRepresentation(value)

    @property
    def numerator(self):
        return self._value.numerator

    @property
    def denominator(self):
        return self._value.denominator

    def __round__(self, *args, **kwargs):
        return self._value.__round__(*args, **kwargs)

    def __abs__(self):
        return TimeType(self._value.__abs__())

    def __hash__(self):
        return self._value.__hash__()

    def __ceil__(self):
        return int(self._value.__ceil__())

    def __floor__(self):
        return int(self._value.__floor__())

    @_with_other_as_time_type
    def __mod__(self, other: 'TimeType'):
        return self._value.__mod__(other._value)

    @_with_other_as_time_type
    def __rmod__(self, other: 'TimeType'):
        return self._value.__rmod__(other._value)

    def __neg__(self):
        return TimeType(self._value.__neg__())

    def __pos__(self):
        return self

    @_with_other_as_time_type
    def __pow__(self, other: 'TimeType'):
        return self._value.__pow__(other._value)

    @_with_other_as_time_type
    def __rpow__(self, other: 'TimeType'):
        return self._value.__rpow__(other._value)

    def __trunc__(self):
        return int(self._value.__trunc__())

    @_with_other_as_time_type
    def __mul__(self, other: 'TimeType'):
        return self._value.__mul__(other._value)

    @_with_other_as_time_type
    def __rmul__(self, other: 'TimeType'):
        return self._value.__mul__(other._value)

    @_with_other_as_time_type
    def __add__(self, other: 'TimeType'):
        return self._value.__add__(other._value)

    @_with_other_as_time_type
    def __radd__(self, other: 'TimeType'):
        return self._value.__radd__(other._value)

    @_with_other_as_time_type
    def __sub__(self, other: 'TimeType'):
        return self._value.__sub__(other._value)

    @_with_other_as_time_type
    def __rsub__(self, other: 'TimeType'):
        return self._value.__rsub__(other._value)

    @_with_other_as_time_type
    def __truediv__(self, other: 'TimeType'):
        return self._value.__truediv__(other._value)

    @_with_other_as_time_type
    def __rtruediv__(self, other: 'TimeType'):
        return self._value.__rtruediv__(other._value)

    @_with_other_as_time_type
    def __floordiv__(self, other: 'TimeType'):
        return self._value.__floordiv__(other._value)

    @_with_other_as_time_type
    def __rfloordiv__(self, other: 'TimeType'):
        return self._value.__rfloordiv__(other._value)

    @_with_other_as_time_type
    def __le__(self, other: 'TimeType'):
        return self._value.__le__(other._value)

    @_with_other_as_time_type
    def __ge__(self, other: 'TimeType'):
        return self._value.__ge__(other._value)

    @_with_other_as_time_type
    def __lt__(self, other: 'TimeType'):
        return self._value.__lt__(other._value)

    @_with_other_as_time_type
    def __gt__(self, other: 'TimeType'):
        return self._value.__gt__(other._value)

    def __eq__(self, other):
        if type(other) == type(self):
            return self._value.__eq__(other._value)
        else:
            return self._value == other

    @classmethod
    def from_float(cls, value: float, absolute_error: typing.Optional[float] = None) -> 'TimeType':
        """Convert a floating point number to a TimeType using one of three modes depending on `absolute_error`.

        absolute_error is None: Use `str(value)` as a proxy to get consistent precision (see below)
        absolute_error == 0: Return the exact value of the float. 0.8 == 3602879701896397 / 4503599627370496
        else: Use absolute error to limit the denominator

        str(value) guarantees that all floats have a different result with sensible rounding.this was chosen as a
        default because it is the expected behaviour most of the time.
        """
        # gmpy2 is at least an order of magnitude faster than fractions.Fraction
        if absolute_error is None:
            # this method utilizes the 'print as many digits as necessary to destinguish between all floats'
            # functionality of str
            return cls(cls._InternalRepresentation(str(value).replace('e', 'E')))

        elif absolute_error == 0:
            return cls(cls._InternalRepresentation(value))
        else:
            if cls._InternalRepresentation is fractions.Fraction:
                return fractions.Fraction(value).limit_denominator(int(1 / absolute_error))
            else:
                return cls(gmpy2.mpq(gmpy2.f2q(value, absolute_error)))

    @classmethod
    def from_fraction(cls, numerator: int, denominator: int) -> 'TimeType':
        return cls(cls._InternalRepresentation(numerator, denominator))

    def __repr__(self):
        return 'TimeType(%s)' % self.__str__()

    def __str__(self):
        return '%d/%d' % (self._value.numerator, self._value.denominator)

    def __float__(self):
        return int(self._value.numerator) / int(self._value.denominator)


_converter = {
    float: TimeType.from_float,
    TimeType: lambda x: x
}


def time_from_float(value: float, absolute_error: typing.Optional[float] = None) -> TimeType:
    warnings.warn("time_from_float is deprecated. Use TimeType.from_float instead", DeprecationWarning)
    return TimeType.from_float(value, absolute_error)


def time_from_fraction(numerator: int, denominator: int) -> TimeType:
    warnings.warn("time_from_fraction is deprecated. Use TimeType.from_fraction instead", DeprecationWarning)
    return TimeType.from_fraction(numerator, denominator)


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
