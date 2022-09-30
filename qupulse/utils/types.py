import typing
import abc
import inspect
import numbers
import fractions
import functools
import warnings
import collections
import operator

import numpy
import sympy

try:
    from frozendict import frozendict
except ImportError:
    warnings.warn("The frozendict package is not installed. We currently also ship a fallback frozendict which "
                  "will be removed in a future release.", category=DeprecationWarning)
    frozendict = None

import qupulse.utils.numeric as qupulse_numeric

__all__ = ["MeasurementWindow", "ChannelID", "HashableNumpyArray", "TimeType", "time_from_float", "DocStringABCMeta",
           "SingletonABCMeta", "SequenceProxy"]

MeasurementWindow = typing.Tuple[str, numbers.Real, numbers.Real]
ChannelID = typing.Union[str, int]

try:
    import gmpy2
    qupulse_numeric.FractionType = gmpy2.mpq

except ImportError:
    gmpy2 = None

    warnings.warn('gmpy2 not found. Using fractions.Fraction as fallback. Install gmpy2 for better performance.'
                  'time_from_float might produce slightly different results')


def _with_other_as_time_type(fn):
    """This is decorator to convert the other argument and the result into a :class:`TimeType`"""
    @functools.wraps(fn)
    def wrapper(self, other) -> 'TimeType':
        converted = _converter.get(type(other), TimeType._try_from_any)(other)
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

    def __init__(self, value: typing.Union[numbers.Rational, int] = 0., denominator: typing.Optional[int] = None):
        """
        Args:
            value: interpreted as Rational if denominator is None. interpreted as numerator otherwise
            denominator: Denominator of the Fraction if not None
        """
        if denominator is not None:
            self._value = self._to_internal(value, denominator)
        elif type(getattr(value, '_value', None)) is self._InternalType:
            self._value = value._value
        else:
            try:
                self._value = self._to_internal(value)
            except TypeError as err:
                raise TypeError(f'Could not create TimeType from {value} of type {type(value)}') from err

    @classmethod
    def _try_from_any(cls, any: typing.Any):
        try:
            cls(any)
        except TypeError:
            pass

        # duck type rational
        if hasattr(any, 'numerator') and hasattr(any, 'denominator'):
            # sympy.Rational has callables...
            numerator = any.numerator() if callable(any.numerator) else any.numerator
            denominator = any.denominator() if callable(any.denominator) else any.denominator
            return cls.from_fraction(int(numerator), int(denominator))

        # test if objects subclass number
        if isinstance(any, numbers.Integral):
            return cls.from_fraction(int(any), 1)
        if isinstance(any, numbers.Real):
            return cls.from_float(float(any))

        # test for array
        if isinstance(any, numpy.ndarray):
            return numpy.vectorize(cls._try_from_any)(any)

        # try conversion to int and float. gmpy2's answer to isinstance is version dependent
        try:
            as_int = int(any)
        except (TypeError, ValueError, RuntimeError):
            as_int = None
        try:
            as_float = float(any)
        except (TypeError, ValueError, RuntimeError):
            as_float = None

        if as_int is None and as_float is not None:
            return cls.from_float(as_float)
        elif as_int is not None and as_float is None:
            return cls.from_fraction(as_int, 1)
        elif as_int is not None and as_float is not None:
            if as_float.is_integer():
                return cls.from_fraction(as_int, 1)
            elif int(as_float) == as_int:
                return cls.from_float(as_float)

        # for error message
        return cls(any)

    @property
    def numerator(self):
        return self._value.numerator

    @property
    def denominator(self):
        return self._value.denominator

    def _sympy_(self):
        import sympy
        return sympy.Rational(self.numerator, self.denominator)

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

    def __le__(self, other):
        return self._value <= self.as_comparable(other)

    def __ge__(self, other):
        return self._value >= self.as_comparable(other)

    def __lt__(self, other):
        return self._value < self.as_comparable(other)

    def __gt__(self, other):
        return self._value > self.as_comparable(other)

    def __eq__(self, other):
        if type(other) == type(self):
            return self._value.__eq__(other._value)
        else:
            return self._value == other

    @classmethod
    def as_comparable(cls, other: typing.Union['TimeType', typing.Any]):
        if type(other) is cls:
            return other._value
        else:
            return other

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
                - 0 < `absolute_error` <= 1: Return the best approximation to `value` within
                  `(value - absolute_error, value + absolute_error)`.
                  The best approximation is defined as the fraction with the smallest denominator.

        Raises:
            ValueError: If `absolute_error` is not None and not 0 <= `absolute_error` <=  1
        """
        # gmpy2 is at least an order of magnitude faster than fractions.Fraction
        if absolute_error is None:
            # this method utilizes the 'print as many digits as necessary to distinguish between all floats'
            # functionality of str
            if type(value) in (cls, cls._InternalType, fractions.Fraction):
                return cls(value)
            else:
                try:
                    # .upper() is a bit faster than replace('e', 'E') which gmpy2.mpq needs
                    return cls(cls._to_internal(str(value).upper()))
                except ValueError:
                    if isinstance(value, numbers.Number) and not numpy.isfinite(value):
                        raise ValueError('Cannot represent "{}" as TimeType'.format(value), value)
                    else:
                        raise

        elif absolute_error == 0:
            return cls(cls._to_internal(value))
        elif absolute_error < 0:
            raise ValueError('absolute_error needs to be > 0')
        elif absolute_error > 1:
            raise ValueError('absolute_error needs to be <= 1')
        else:
            return cls(qupulse_numeric.approximate_double(value, absolute_error, fraction_type=cls._to_internal))

    @classmethod
    def from_fraction(cls, numerator: int, denominator: int) -> 'TimeType':
        """Convert a fraction to a TimeType.

        Args:
            numerator: Numerator of the time fraction
            denominator: Denominator of the time fraction
        """
        return cls(numerator, denominator)

    def __repr__(self):
        return f'TimeType({self._value.numerator}, {self._value.denominator})'

    def __str__(self):
        return '%d/%d' % (self._value.numerator, self._value.denominator)

    def __float__(self):
        return int(self._value.numerator) / int(self._value.denominator)


# this asserts isinstance(TimeType, Rational) is True
numbers.Rational.register(TimeType)


_converter = {
    float: TimeType.from_float,
    TimeType._InternalType: TimeType,
    fractions.Fraction: TimeType,
    sympy.Rational: lambda q: TimeType.from_fraction(q.p, q.q),
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

    Deprecated since 0.6. This is a bad idea.

    Example usage:
    my_array = np.zeros([1, 2, 3, 4])
    hashable = my_array.view(HashableNumpyArray)
    """
    def __array_finalize__(self, obj):
        warnings.warn("HashableNumpyArray is deprecated since qupulse 0.6 and will be removed in the next release.",
                      category=DeprecationWarning, stacklevel=2)

    def __hash__(self):
        return hash(self.tobytes())


@functools.lru_cache(maxsize=128)
def _public_type_attributes(type_obj):
    return {attr for attr in dir(type_obj) if not attr.startswith('_')}

def has_type_interface(obj: typing.Any, type_obj: typing.Type) -> bool:
    """Return true if all public attributes of the class are attributes of the object"""
    return set(dir(obj)) >= _public_type_attributes(type_obj)


_KT_hash = typing.TypeVar('_KT_hash', bound=typing.Hashable)  # Key type.
_T_co_hash = typing.TypeVar('_T_co_hash', bound=typing.Hashable, covariant=True)  # Any type covariant containers.

FrozenMapping = typing.Mapping[_KT_hash, _T_co_hash]


class _FrozenDictByInheritance(dict):
    """This is non mutable, hashable dict. It violates the Liskov substitution principle but is faster than wrapping.
    It is not used by default and may be removed in the future.
    """
    def __setitem__(self, key, value):
        raise TypeError('FrozenDict is immutable')

    def __delitem__(self, key):
        raise TypeError('FrozenDict is immutable')

    def update(self, *args, **kwargs):
        raise TypeError('FrozenDict is immutable')

    def setdefault(self, *args, **kwargs):
        raise TypeError('FrozenDict is immutable')

    def clear(self):
        raise TypeError('FrozenDict is immutable')

    def pop(self, *args, **kwargs):
        raise TypeError('FrozenDict is immutable')

    def popitem(self, *args, **kwargs):
        raise TypeError('FrozenDict is immutable')

    def copy(self):
        return self

    def to_dict(self) -> typing.Dict[_KT_hash, _T_co_hash]:
        return super().copy()

    def __hash__(self):
        # faster than functools.reduce(operator.xor, map(hash, self.items())) but takes more memory
        # TODO: investigate caching
        return hash(frozenset(self.items()))


class _FrozenDictByWrapping(FrozenMapping):
    """Immutable dict like type.

    There are the following possibilities in pure python:
     - subclass dict (violates the Liskov substitution principle)
     - wrap dict (slow construction and method indirection)
     - abuse MappingProxyType (hard to add hash and make mutation difficult)



    Wrapper around builtin dict without the mutating methods.

    Hot path methods in __slots__ are the bound methods of the dict object. The other methods are wrappers.

    Why not subclass dict and overwrite mutating methods:
        roughly the same speed for __slot__ methods (a bit slower than native dict)
        dict subclass always implements MutableMapping which makes type annotations useless
        caching the hash value is slightly slower for the subclass

    Only downside: This wrapper class needs to implement __init__ and copy the __slot__ methods which is an overhead of
                    ~10 i.e. 250ns for empty subclass init vs. 4µs for empty wrapper init
    """
    # made concessions in code style due to performance
    _HOT_PATH_METHODS = ('keys', 'items', 'values', 'get', '__getitem__')
    _PRIVATE_ATTRIBUTES = ('_hash', '_dict')
    __slots__ = _HOT_PATH_METHODS + _PRIVATE_ATTRIBUTES

    def __new__(cls, *args, **kwds):
        """Overwriting __new__ saves a factor of two for initialization. This is the relevant line from
        Generic.__new__"""
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        inner_dict = dict(*args, **kwargs)
        self._dict = inner_dict  # type: typing.Dict[_KT_hash, _T_co_hash]
        self._hash = None

        self.__getitem__ = inner_dict.__getitem__
        self.keys = inner_dict.keys
        self.items = inner_dict.items
        self.values = inner_dict.values
        self.get = inner_dict.get

    def __contains__(self, item: _KT_hash) -> bool:
        return item in self._dict

    def __iter__(self) -> typing.Iterator[_KT_hash]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self._dict)

    def __hash__(self) -> int:
        # use the local variable h to minimize getattr calls to minimum and reduce caching overhead
        h = self._hash
        if h is None:
            self._hash = h = functools.reduce(operator.xor, map(hash, self.items()), 0xABCD0)
        return h

    def __eq__(self, other: typing.Mapping):
        return other == self._dict

    def copy(self):
        return self

    def to_dict(self) -> typing.Dict[_KT_hash, _T_co_hash]:
        return self._dict.copy()


if frozendict is None:
    FrozenDict = _FrozenDictByWrapping
else:
    FrozenDict = frozendict


class SequenceProxy(collections.abc.Sequence):
    __slots__ = ('_inner',)

    def __init__(self, inner: typing.Sequence):
        self._inner = inner

    def __getitem__(self, item):
        return self._inner.__getitem__(item)

    def __iter__(self):
        return self._inner.__iter__()

    def __len__(self):
        return self._inner.__len__()

    def __contains__(self, item):
        return self._inner.__contains__(item)

    def __reversed__(self):
        return self._inner.__reversed__()

    def index(self, i, **kwargs):
        return self._inner.index(i, **kwargs)

    def count(self, elem):
        return self._inner.count(elem)

    def __eq__(self, other):
        """Not part of Sequence interface"""
        if type(other) is SequenceProxy:
            return (len(self) == len(other)
                    and all(x == y for x, y in zip(self, other)))
        else:
            return NotImplemented


