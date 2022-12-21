"""This module defines parameter constriants.
"""

from abc import abstractmethod
from typing import Optional, Union, Dict, Any, Iterable, Set, List, Mapping, AbstractSet
from numbers import Real
import warnings

import sympy
import numpy

from qupulse.serialization import AnonymousSerializable
from qupulse.expressions import Expression
from qupulse.parameter_scope import Scope, ParameterNotProvidedException

__all__ = ["ParameterNotProvidedException", "ParameterConstraintViolation", "ParameterConstraint"]


class ParameterConstraint(AnonymousSerializable):
    """A parameter constraint like 't_2 < 2.7' that can be used to set bounds to parameters."""
    def __init__(self, relation: Union[str, sympy.Expr]):
        super().__init__()
        if isinstance(relation, str) and '==' in relation:
            # The '==' operator is interpreted by sympy as exactly, however we need a symbolical evaluation
            self._expression = sympy.Eq(*sympy.sympify(relation.split('==')))
        else:
            self._expression = sympy.sympify(relation)
        if not isinstance(self._expression, sympy.logic.boolalg.Boolean):
            raise ValueError('Constraint is not boolean')
        self._expression = Expression(self._expression)

    @property
    def affected_parameters(self) -> Set[str]:
        return set(self._expression.variables)

    def is_fulfilled(self, parameters: Mapping[str, Any], volatile: AbstractSet[str] = frozenset()) -> bool:
        """
        Args:
            parameters: These parameters are checked.
            volatile: For each of these parameters a warning is raised if they appear in a constraint

        Raises:
            :class:`qupulse.parameter_scope.ParameterNotProvidedException`: if a parameter is missing

        Warnings:
            ConstrainedParameterIsVolatileWarning: if a constrained parameter is volatile
        """
        affected_parameters = self.affected_parameters
        if not affected_parameters.issubset(parameters.keys()):
            raise ParameterNotProvidedException((affected_parameters-parameters.keys()).pop())

        for parameter in volatile & affected_parameters:
            warnings.warn(ConstrainedParameterIsVolatileWarning(parameter_name=parameter, constraint=self))

        return numpy.all(self._expression.evaluate_in_scope(parameters))

    @property
    def sympified_expression(self) -> sympy.Expr:
        return self._expression.underlying_expression

    def __eq__(self, other: 'ParameterConstraint') -> bool:
        return self._expression.underlying_expression == other._expression.underlying_expression

    def __str__(self) -> str:
        if isinstance(self._expression.underlying_expression, sympy.Eq):
            return '{}=={}'.format(self._expression.underlying_expression.lhs,
                                   self._expression.underlying_expression.rhs)
        else:
            return str(self._expression.underlying_expression)

    def __repr__(self):
        return 'ParameterConstraint(%s)' % repr(str(self))

    def get_serialization_data(self) -> str:
        return str(self)


ConstraintLike = Union[sympy.Expr, str, ParameterConstraint]


class ParameterConstrainer:
    """A class that implements the testing of parameter constraints. It is used by the subclassing pulse templates."""
    def __init__(self, *,
                 parameter_constraints: Optional[Iterable[ConstraintLike]]) -> None:
        if parameter_constraints is None:
            self._parameter_constraints = []
        else:
            self._parameter_constraints = [constraint if isinstance(constraint, ParameterConstraint)
                                           else ParameterConstraint(constraint)
                                           for constraint in parameter_constraints]

    @property
    def parameter_constraints(self) -> List[ParameterConstraint]:
        return self._parameter_constraints

    def validate_parameter_constraints(self, parameters: [str, Real], volatile: Set[str]) -> None:
        """
        Raises a ParameterConstraintViolation exception if one of the constraints is violated.

        Args:
            parameters: These parameters are checked.
            volatile: For each of these parameters a warning is raised if they appear in a constraint

        Raises:
            ParameterConstraintViolation: if one of the constraints is violated.

        Warnings:
            ConstrainedParameterIsVolatileWarning: via `ParameterConstraint.is_fulfilled`
        """
        for constraint in self._parameter_constraints:
            if not constraint.is_fulfilled(parameters, volatile=volatile):
                raise ParameterConstraintViolation(constraint, parameters)

    def validate_scope(self, scope: Scope):
        volatile = scope.get_volatile_parameters().keys()

        for constraint in self._parameter_constraints:
            if not constraint.is_fulfilled(scope, volatile=volatile):
                constrained_parameters = {parameter_name: scope[parameter_name]
                                          for parameter_name in constraint.affected_parameters}
                raise ParameterConstraintViolation(constraint, constrained_parameters)

    @property
    def constrained_parameters(self) -> AbstractSet[str]:
        return set().union(*(c.affected_parameters for c in self._parameter_constraints))


class ParameterConstraintViolation(Exception):
    def __init__(self, constraint: ParameterConstraint, parameters: Dict[str, Real]):
        super().__init__("The constraint '{}' is not fulfilled.\nParameters: {}".format(constraint, parameters))
        self.constraint = constraint
        self.parameters = parameters


class InvalidParameterNameException(Exception):
    def __init__(self, parameter_name: str):
        self.parameter_name = parameter_name

    def __str__(self) -> str:
        return '{} is an invalid parameter name'.format(self.parameter_name)


class ConstrainedParameterIsVolatileWarning(RuntimeWarning):
    def __init__(self, parameter_name: str, constraint: ParameterConstraint):
        super().__init__(parameter_name, constraint)

    @property
    def parameter_name(self) -> str:
        return self.args[0]

    @property
    def constraint(self) -> ParameterConstraint:
        return self.args[1]

    def __str__(self):
        return ("The parameter '{parameter_name}' is constrained "
                "by '{constraint}' but marked as volatile").format(parameter_name=self.parameter_name,
                                                                   constraint=self.constraint)
