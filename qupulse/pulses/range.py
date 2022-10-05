from typing import Tuple, Any, AbstractSet, Mapping, Union, Iterator
from numbers import Number
from dataclasses import dataclass
from functools import lru_cache
import itertools

from qupulse.utils import checked_int_cast, cached_property
from qupulse.expressions import ExpressionScalar, ExpressionVariableMissingException, ExpressionLike, Expression
from qupulse.parameter_scope import Scope
from qupulse.utils.types import FrozenDict, FrozenMapping

RangeLike = Union[range,
                  ExpressionLike,
                  Tuple[ExpressionLike, ExpressionLike],
                  Tuple[ExpressionLike, ExpressionLike, ExpressionLike]]


@dataclass(frozen=True)
class ParametrizedRange:
    start: ExpressionScalar
    stop: ExpressionScalar
    step: ExpressionScalar

    def __init__(self, *args, **kwargs):
        """Like the builtin python range but with parameters. Positional and keyword arguments cannot be mixed.

        Args:
            *args: Interpreted as ``(start, )`` or ``(start, stop[, step])``
            **kwargs: Expected to contain ``start``, ``stop`` and ``step``
        Raises:
            TypeError: If positional and keyword arguments are mixed
            KeyError: If keyword arguments but one of ``start``, ``stop`` or ``step`` is missing
        """
        if args and kwargs:
            raise TypeError('ParametrizedRange only takes either positional or keyword arguments')
        elif kwargs:
            start = kwargs['start']
            stop = kwargs['stop']
            step = kwargs['step']
        elif len(args) in (1, 2, 3):
            if len(args) == 3:
                start, stop, step = args
            elif len(args) == 2:
                (start, stop), step = args, 1
            else:
                start, (stop,), step = 0, args, 1
        else:
            raise TypeError('ParametrizedRange expected 1 to 3 arguments, got {}'.format(len(args)), args)

        object.__setattr__(self, 'start', ExpressionScalar(start))
        object.__setattr__(self, 'stop', ExpressionScalar(stop))
        object.__setattr__(self, 'step', ExpressionScalar(step))

    @lru_cache(maxsize=1024)
    def to_tuple(self) -> Tuple[Any, Any, Any]:
        """Return a simple representation of the range which is useful for comparison and serialization"""
        return (self.start.get_serialization_data(),
                self.stop.get_serialization_data(),
                self.step.get_serialization_data())

    def to_range(self, parameters: Mapping[str, Number]) -> range:
        return range(checked_int_cast(self.start.evaluate_in_scope(parameters)),
                     checked_int_cast(self.stop.evaluate_in_scope(parameters)),
                     checked_int_cast(self.step.evaluate_in_scope(parameters)))

    @cached_property
    def parameter_names(self) -> AbstractSet[str]:
        return set(self.start.variables) | set(self.stop.variables) | set(self.step.variables)

    @classmethod
    def from_range_like(cls, range_like: RangeLike):
        if isinstance(range_like, cls):
            return range_like
        elif isinstance(range_like, (tuple, list)):
            return cls(*range_like)
        elif isinstance(range_like, range):
            return cls(range_like.start, range_like.stop, range_like.step)
        elif isinstance(range_like, slice):
            raise TypeError("Cannot construct a range from a slice")
        else:
            return cls(range_like)

    def get_serialization_data(self):
        return self.to_tuple()


class RangeScope(Scope):
    __slots__ = ('_index_name', '_index_value', '_inner')

    def __init__(self, inner: Scope, index_name: str, index_value: int):
        super().__init__()
        self._inner = inner
        self._index_name = index_name
        self._index_value = index_value

    def get_volatile_parameters(self) -> FrozenMapping[str, Expression]:
        inner_volatile = self._inner.get_volatile_parameters()

        if self._index_name in inner_volatile:
            # TODO: use delete method of frozendict
            index_name = self._index_name
            return FrozenDict((name, value) for name, value in inner_volatile.items() if name != index_name)
        else:
            return inner_volatile

    def __hash__(self):
        return hash((self._inner, self._index_name, self._index_value))

    def __eq__(self, other: 'RangeScope'):
        try:
            return (self._index_name == other._index_name
                    and self._index_value == other._index_value
                    and self._inner == other._inner)
        except AttributeError:
            return NotImplemented

    def __contains__(self, item):
        return item == self._index_name or item in self._inner

    def get_parameter(self, parameter_name: str) -> Number:
        if parameter_name == self._index_name:
            return self._index_value
        else:
            return self._inner.get_parameter(parameter_name)

    __getitem__ = get_parameter

    def change_constants(self, new_constants: Mapping[str, Number]) -> 'Scope':
        return RangeScope(self._inner.change_constants(new_constants), self._index_name, self._index_value)

    def __len__(self) -> int:
        return len(self._inner) + int(self._index_name not in self._inner)

    def __iter__(self) -> Iterator:
        if self._index_name in self._inner:
            return iter(self._inner)
        else:
            return itertools.chain(self._inner, (self._index_name,))

    def as_dict(self) -> FrozenMapping[str, Number]:
        if self._as_dict is None:
            self._as_dict = FrozenDict({**self._inner.as_dict(), self._index_name: self._index_value})
        return self._as_dict

    def keys(self):
        return self.as_dict().keys()

    def items(self):
        return self.as_dict().items()

    def values(self):
        return self.as_dict().values()

    def __repr__(self):
        return f'{type(self)}(inner={self._inner!r}, index_name={self._index_name!r}, ' \
               f'index_value={self._index_value!r})'
