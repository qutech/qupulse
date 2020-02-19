from abc import abstractmethod
from typing import Optional, Union, Dict, Any, Iterable, Set, List, Mapping, AbstractSet
from numbers import Number
import functools
import collections
import itertools
import warnings
import operator

import sympy
import numpy

from qupulse.serialization import AnonymousSerializable
from qupulse.expressions import Expression, ExpressionVariableMissingException
from qupulse.utils.types import HashableNumpyArray, DocStringABCMeta, Collection, SingletonABCMeta, FrozenMapping,\
    FrozenDict


class Scope(Mapping[str, Number]):
    __slots__ = ()

    @abstractmethod
    def get_volatile_parameters(self) -> AbstractSet[str]:
        pass

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    def __getitem__(self, item):
        return self.get_parameter(item)

    @abstractmethod
    def get_parameter(self, parameter_name: str) -> Number:
        """
        Args:
            parameter_name:

        Raises:
            ParameterNotProvidedException if the parameter is not provided by this scope

        Returns:
            Parameter value
        """

    @abstractmethod
    def change_constants(self, new_constants: Mapping[str, Number]) -> 'Scope':
        """Change values of constants. Constants not present in the scope are ignored.

        Args:
            new_constants:

        Warnings:
            NonVolatileChange: if a parameter that is not in get_volatile_parameters is updated

        Returns:
            New scope instance
        """

    def overwrite(self, to_overwrite: Mapping[str, Number]) -> 'Scope':
        """"""
        # TODO: replace with OverwritingScope
        return MappedScope(self, FrozenDict((name, Expression(value))
                                            for name, value in to_overwrite.items()))


class MappedScope(Scope):
    __slots__ = ('_scope', '_mapping', '_cache', '_volatile_parameters_cache')

    def __init__(self, scope: Scope, mapping: FrozenMapping[str, Expression]):
        super(MappedScope, self).__init__()
        self._scope = scope
        self._mapping = mapping
        self._volatile_parameters_cache = None

        self._cache = {}
        # TODO use caching

    def keys(self) -> AbstractSet[str]:
        return self._scope.keys() | self._mapping.keys()

    def __contains__(self, item):
        return item in self._mapping or item in self._scope

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

    def __repr__(self):
        return '%s(scope=%r, mapping=%r)' % (self.__class__.__name__, self._scope, self._mapping)

    def _calc_parameter(self, parameter_name: str) -> Number:
        expression = self._mapping.get(parameter_name, None)
        if expression is None:
            return self._scope.get_parameter(parameter_name)
        else:
            try:
                return expression.evaluate_in_scope(self._scope)
            except ExpressionVariableMissingException as err:
                raise ParameterNotProvidedException(err.variable) from err

    def get_parameter(self, parameter_name: str) -> Number:
        result = self._cache.get(parameter_name, None)
        if result is None:
            self._cache[parameter_name] = result = self._calc_parameter(parameter_name)
        return result

    def __hash__(self):
        return hash((self._scope, self._mapping))

    def __eq__(self, other: 'MappedScope'):
        return self._scope == other._scope and self._mapping == other._mapping

    def change_constants(self, new_constants: Mapping[str, Number]) -> 'Scope':
        scope = self._scope.change_constants(new_constants)
        if scope is self._scope:
            return self
        else:
            return MappedScope(
                scope=scope,
                mapping=self._mapping
            )

    def _collect_volatile_parameters(self) -> AbstractSet[str]:
        inner_volatile = self._scope.get_volatile_parameters()
        if inner_volatile:
            non_volatile = set()
            volatile = set()
            for mapped_parameter, expression in self._mapping.items():
                if inner_volatile.isdisjoint(expression.variables):
                    non_volatile.add(mapped_parameter)
                else:
                    volatile.add(mapped_parameter)

            result = inner_volatile - non_volatile
            result |= volatile
            return frozenset(result)
        else:
            return inner_volatile

    def get_volatile_parameters(self) -> AbstractSet[str]:
        if self._volatile_parameters_cache is None:
            self._volatile_parameters_cache = self._collect_volatile_parameters()
        return self._volatile_parameters_cache


class DictScope(Scope):
    __slots__ = ('_values', '_volatile_parameters', 'keys', 'items', 'values')

    def __init__(self, values: FrozenMapping[str, Number], volatile: AbstractSet[str] = frozenset()):
        super().__init__()
        assert getattr(values, '__hash__', None) is not None

        self._values = values
        self._volatile_parameters = frozenset(volatile)
        self.keys = self._values.keys
        self.items = self._values.items
        self.values = self._values.values

    def __contains__(self, parameter_name):
        return parameter_name in self._values

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        return '%s(values=%r)' % (self.__class__.__name__, self._values)

    def get_parameter(self, parameter_name) -> Number:
        try:
            return self._values[parameter_name]
        except KeyError:
            raise ParameterNotProvidedException(parameter_name)

    def __hash__(self):
        return hash((self._values, self._volatile_parameters))

    def __eq__(self, other: 'DictScope'):
        if type(self) is type(other):
            return self._values == other._values and self._volatile_parameters == other._volatile_parameters
        else:
            return NotImplemented

    def change_constants(self, new_constants: Mapping[str, Number]) -> 'Scope':
        to_update = new_constants.keys() & self._values.keys()
        if to_update:
            updated_non_volatile = to_update - self.get_volatile_parameters()
            if updated_non_volatile:
                warnings.warn(NonVolatileChange(updated_non_volatile))

            return DictScope(
                values=FrozenDict((parameter_name, new_constants.get(parameter_name, old_value))
                                  for parameter_name, old_value in self._values.items()),
                volatile=self._volatile_parameters
            )
        else:
            return self

    def get_volatile_parameters(self) -> AbstractSet[str]:
        return self._volatile_parameters

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Number], volatile: AbstractSet[str] = frozenset()) -> 'DictScope':
        return cls(FrozenDict(mapping), volatile)

    @classmethod
    def from_kwargs(cls, *, volatile: AbstractSet[str] = frozenset(), **kwargs: Number) -> 'DictScope':
        return cls.from_mapping(kwargs, volatile)


class JointScope(Scope):
    __slots__ = ('_lookup', '_volatile_parameters')

    def __init__(self, lookup: FrozenMapping[str, Scope]):
        self._lookup = lookup
        self._volatile_parameters = None

    def __contains__(self, parameter_name):
        return parameter_name in self._lookup

    def __iter__(self):
        return iter(self._lookup)

    def __len__(self):
        return len(self._lookup)

    def __repr__(self):
        return '%s(lookup=%r)' % (self.__class__.__name__, self._lookup)

    def get_parameter(self, parameter_name: str) -> Number:
        return self._lookup[parameter_name].get_parameter(parameter_name)

    def __hash__(self):
        return hash(frozenset(self.items()))

    def __eq__(self, other: 'JointScope'):
        return frozenset(self.items()) == frozenset(other.items())

    def change_constants(self, new_constants: Mapping[str, Number]) -> 'Scope':
        return JointScope(FrozenDict(
            (parameter_name, scope.change_constants(new_constants)) for parameter_name, scope in self._lookup.items()
        ))

    def get_volatile_parameters(self) -> AbstractSet[str]:
        if self._volatile_parameters is None:
            volatile_parameters = functools.reduce(operator.or_, (scope.get_volatile_parameters()
                                                                  for scope in self._lookup.values()))
            self._volatile_parameters = frozenset(self._lookup.keys() & volatile_parameters)
        return self._volatile_parameters


class ParameterNotProvidedException(KeyError):
    """Indicates that a required parameter value was not provided."""

    def __init__(self, parameter_name: str) -> None:
        super().__init__(parameter_name)

    @property
    def parameter_name(self):
        return self.args[0]

    def __str__(self) -> str:
        return "No value was provided for parameter '{0}'.".format(self.parameter_name)


class NonVolatileChange(RuntimeWarning):
    """Raised if a non volatile parameter is updated"""
