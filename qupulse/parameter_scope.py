from abc import abstractmethod
from typing import Optional, Union, Dict, Any, Iterable, Set, List, Mapping
from numbers import Real
import functools
import collections

import sympy
import numpy

from qupulse.serialization import AnonymousSerializable
from qupulse.expressions import Expression, ExpressionVariableMissingException
from qupulse.utils.types import HashableNumpyArray, DocStringABCMeta, Collection, SingletonABCMeta, FrozenDict


class Scope(metaclass=DocStringABCMeta):
    @abstractmethod
    def get_parameter(self, parameter_name) -> Real:
        pass

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def change_constants(self, new_constants: Mapping[str, Real]) -> 'Scope':
        """Change values of constants"""


class MappedScope(Scope):
    def __init__(self, scope: Scope, mapping: FrozenDict[str, Expression]):
        self._scope = scope
        self._mapping = mapping
        self.get_parameter = functools.lru_cache(maxsize=None)(self.get_parameter)

    def get_parameter(self, parameter_name) -> Real:
        expression = self._mapping.get(parameter_name, None)
        scope_get_parameter = self._scope.get_parameter
        if expression is None:
            return scope_get_parameter(parameter_name)
        else:
            dependencies = {inner_parameter: scope_get_parameter(inner_parameter)
                            for inner_parameter in expression.variables}
            return expression.evaluate_numeric(**dependencies)

    def __hash__(self):
        return hash((self._scope, self._mapping))

    def __eq__(self, other: 'MappedScope'):
        return self._scope == other._scope and self._mapping == other._mapping

    def change_constants(self, new_constants: Mapping[str, Real]) -> 'Scope':
        scope = self._scope.change_constants(new_constants)
        if scope is self._scope:
            return self
        else:
            return MappedScope(
                scope=scope,
                mapping=self._mapping
            )


class DictScope(Scope, metaclass=SingletonABCMeta):
    __slots__ = ('_values',)

    def __init__(self, values: FrozenDict[str, Real]):
        self._values = values

    def get_parameter(self, parameter_name) -> Real:
        return self._values[parameter_name]

    def __hash__(self):
        return hash(self._values)

    def __eq__(self, other: 'DictScope'):
        return self._values == other._values

    def change_constants(self, new_constants: Mapping[str, Real]) -> 'Scope':
        if new_constants.keys() & self._values.keys():
            return DictScope(
                values=FrozenDict((parameter_name, new_constants.get(parameter_name, old_value))
                                  for parameter_name, old_value in self._values.items())
            )
        else:
            return self

