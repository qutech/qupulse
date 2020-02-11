from abc import abstractmethod
from typing import Optional, Union, Dict, Any, Iterable, Set, List, Mapping, AbstractSet
from numbers import Number
import functools
import collections
import itertools
import warnings

import sympy
import numpy

from qupulse.serialization import AnonymousSerializable
from qupulse.expressions import Expression, ExpressionVariableMissingException
from qupulse.utils.types import HashableNumpyArray, DocStringABCMeta, Collection, SingletonABCMeta, FrozenDict


class Scope(Mapping[str, Number]):
    __slots__ = ()

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
        pass

    @abstractmethod
    def change_constants(self, new_constants: Mapping[str, Number]) -> 'Scope':
        """Change values of constants"""


class MappedScope(Scope):
    __slots__ = ('_scope', '_mapping', 'get_parameter')

    def __init__(self, scope: Scope, mapping: FrozenDict[str, Expression]):
        self._scope = scope
        self._mapping = mapping
        self.get_parameter = functools.lru_cache(maxsize=None)(self.get_parameter)

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

    def get_parameter(self, parameter_name: str) -> Number:
        expression = self._mapping.get(parameter_name, None)
        scope_get_parameter = self._scope.get_parameter
        if expression is None:
            return scope_get_parameter(parameter_name)
        else:
            return expression.evaluate_in_scope(self._scope)

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


class DictScope(Scope):
    __slots__ = ('_values', 'keys')

    def __init__(self, values: FrozenDict[str, Number]):
        self._values = values
        self.keys = self._values.keys()

    def __contains__(self, parameter_name):
        return parameter_name in self._values

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        return '%s(values=%r)' % (self.__class__.__name__, self._values)

    def get_parameter(self, parameter_name) -> Number:
        return self._values[parameter_name]

    def __hash__(self):
        return hash(self._values)

    def __eq__(self, other: 'DictScope'):
        return self._values == other._values

    def change_constants(self, new_constants: Mapping[str, Number]) -> 'Scope':
        if new_constants.keys() & self._values.keys():
            return DictScope(
                values=FrozenDict((parameter_name, new_constants.get(parameter_name, old_value))
                                  for parameter_name, old_value in self._values.items())
            )
        else:
            return self
