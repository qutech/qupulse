from typing import NamedTuple, Mapping
import warnings
import numbers


from qupulse.parameter_scope import Scope, MappedScope, JointScope
from qupulse.expressions import Expression, ExpressionScalar
from qupulse.utils.types import FrozenDict, FrozenMapping
from qupulse.utils import is_integer


VolatileProperty = NamedTuple('VolatileProperty', [('expression', Expression),
                                                   ('dependencies', FrozenMapping[str, Expression])])
VolatileProperty.__doc__ = """Hashable representation of a volatile program property. It does not contain the concrete
value. Using the dependencies attribute to calculate the value might yield unexpected results."""


class VolatileValue:
    """Not hashable"""

    def __init__(self, expression: ExpressionScalar, scope: Scope):
        self._expression = expression
        self._scope = scope

    @property
    def volatile_property(self) -> VolatileProperty:
        dependencies = self._scope.get_volatile_parameters()
        dependencies = FrozenDict({parameter_name: dependencies[parameter_name]
                                   for parameter_name in self._expression.variables
                                   if parameter_name in dependencies})
        return VolatileProperty(expression=self._expression, dependencies=dependencies)

    @classmethod
    def operation(cls, expression, **operands):
        expression = Expression(expression)
        assert set(expression.variables) == operands.keys()

        scope = JointScope(FrozenDict(
            {operand_name: MappedScope(operand._scope, FrozenDict({operand_name: operand._expression}))
             for operand_name, operand in operands.items()}
        ))
        return cls(expression, scope)
    
    def __sub__(self, other: int):
        return type(self)(self._expression - other, self._scope)

    def __mul__(self, other: int):
        return type(self)(self._expression * other, self._scope)


class VolatileRepetitionCount(VolatileValue):
    def __int__(self):
        value = self._expression.evaluate_in_scope(self._scope)
        if not is_integer(value):
            warnings.warn("Repetition count is no integer. Rounding might lead to unexpected results.")
        value = int(round(value))
        if value < 0:
            warnings.warn("Repetition count is negative. Clamping lead to unexpected results.")
            value = 0
        return value

    def update_volatile_dependencies(self, new_constants: Mapping[str, numbers.Number]) -> int:
        self._scope = self._scope.change_constants(new_constants)
        return int(self)

    def __eq__(self, other):
        if type(self) == type(other):
            return self._scope is other._scope and self._expression == other._expression
        else:
            return NotImplemented
