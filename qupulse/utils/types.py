import typing
import abc
import inspect
import numbers
import fractions
import functools
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
    """This is decorator to convert the other argument and the result into a :class:`TimeType`"""
    @functools.wraps(fn)
    def wrapper(self, other) -> 'TimeType':
        converted = _converter.get(type(other), TimeType)(other)
        result = fn(self, converted)
        if result is NotImplemented:
            return result
        elif type(result) is TimeType._InternalType:
            return TimeType(result)
        else:
            return result
    return wrapper


class TimeType:
    """This type represents a rational number with arbitrary precision.

    Internally it uses :func:`gmpy2.mpq` (if available) or :class:`fractions.Fraction`
    """
    __slots__ = ('_value',)

    _InternalType = fractions.Fraction if gmpy2 is None else type(gmpy2.mpq())
    _to_internal = fractions.Fraction if gmpy2 is None else gmpy2.mpq

    def __init__(self, value: numbers.Rational = 0.):
        if type(value) == type(self):
            self._value = value._value
        else:
            self._value = self._to_internal(value)

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

    def __int__(self):
        return int(self._value)

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

        The default str(value) guarantees that all floats have a different result with sensible rounding.
        This was chosen as default because it is the expected behaviour most of the time if the user defined the float
        from a literal in code.

        Args:
            value: Floating point value to convert to arbitrary precision TimeType
            absolute_error:
                - :obj:`None`: Use `str(value)` as a proxy to get consistent precision
                - 0: Return the exact value of the float i.e. float(0.8) == 3602879701896397 / 4503599627370496
                - 0 < `absolute_error` <= 1: Use `absolute_error` to limit the denominator

        Raises:
            ValueError: If `absolute_error` is not None and not 0 <= `absolute_error` <=  1
        """
        # gmpy2 is at least an order of magnitude faster than fractions.Fraction
        if absolute_error is None:
            # this method utilizes the 'print as many digits as necessary to destinguish between all floats'
            # functionality of str
            if type(value) in (cls, cls._InternalType, fractions.Fraction):
                return cls(value)
            else:
                return cls(cls._to_internal(str(value).replace('e', 'E')))

        elif absolute_error == 0:
            return cls(cls._to_internal(value))
        elif absolute_error < 0:
            raise ValueError('absolute_error needs to be at least 0')
        elif absolute_error > 1:
            raise ValueError('absolute_error needs to be smaller 1')
        else:
            if cls._InternalType is fractions.Fraction:
                return fractions.Fraction(value).limit_denominator(int(1 / absolute_error))
            else:
                return cls(gmpy2.f2q(value, absolute_error))

    @classmethod
    def from_fraction(cls, numerator: int, denominator: int) -> 'TimeType':
        """Convert a fraction to a TimeType.

        Args:
            numerator: Numerator of the time fraction
            denominator: Denominator of the time fraction
        """
        return cls(cls._to_internal(numerator, denominator))

    def __repr__(self):
        return 'TimeType(%s)' % self.__str__()

    def __str__(self):
        return '%d/%d' % (self._value.numerator, self._value.denominator)

    def __float__(self):
        return int(self._value.numerator) / int(self._value.denominator)


# this asserts isinstance(TimeType, Rational) is True
numbers.Rational.register(TimeType)


_converter = {
    float: TimeType.from_float,
    TimeType: lambda x: x
}


def time_from_float(value: float, absolute_error: typing.Optional[float] = None) -> TimeType:
    """See :func:`TimeType.from_float`."""
    return TimeType.from_float(value, absolute_error)


def time_from_fraction(numerator: int, denominator: int) -> TimeType:
    """See :func:`TimeType.from_float`."""
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
