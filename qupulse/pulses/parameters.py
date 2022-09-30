"""This module defines parameters and parameter declaration for usage in pulse modelling.

Classes:
    - Parameter: A base class representing a single pulse parameter.
    - ConstantParameter: A single parameter with a constant value.
    - MappedParameter: A parameter whose value is mathematically computed from another parameter.
    - ParameterNotProvidedException.
"""

from abc import abstractmethod
from typing import Optional, Union, Dict, Any, Iterable, Set, List, Mapping, AbstractSet
from numbers import Real
import warnings

import sympy
import numpy

from qupulse.serialization import AnonymousSerializable
from qupulse.expressions import Expression, ExpressionVariableMissingException
from qupulse.parameter_scope import Scope, ParameterNotProvidedException
from qupulse.utils.types import HashableNumpyArray, DocStringABCMeta

__all__ = ["Parameter", "ConstantParameter",
           "ParameterNotProvidedException", "ParameterConstraintViolation", "ParameterConstraint"]


class Parameter(metaclass=DocStringABCMeta):
    """A parameter for pulses.
    
    Parameter specifies a concrete value which is inserted instead
    of the parameter declaration reference in a PulseTemplate if it satisfies
    the minimum and maximum boundary of the corresponding ParameterDeclaration.
    Implementations of Parameter may provide a single constant value or
    obtain values by computation (e.g. from measurement results).
    """
    @abstractmethod
    def get_value(self) -> Real:
        """Compute and return the parameter value."""

    @property
    @abstractmethod
    def requires_stop(self) -> bool:
        """Query whether the evaluation of this Parameter instance requires an interruption in
        execution/sequencing, e.g., because it depends on data that is only measured in during the
        next execution.

        Returns:
            True, if evaluating this Parameter instance requires an interruption.
        """

    def __eq__(self, other: 'Parameter') -> bool:
        return numpy.array_equal(self.get_value(), other.get_value())
        
        
class ConstantParameter(Parameter):
    def __init__(self, value: Union[Real, numpy.ndarray, Expression, str, sympy.Expr]) -> None:
        """
        .. deprecated:: 0.5

        A pulse parameter with a constant value.

        Args:
            value: The value of the parameter
        """
        warnings.warn("ConstantParameter is deprecated. Use plain number types instead", DeprecationWarning,
                      stacklevel=2)

        super().__init__()
        try:
            if isinstance(value, Real):
                self._value = value
            elif isinstance(value, (str, Expression, sympy.Expr)):
                self._value = Expression(value).evaluate_numeric()
            else:
                self._value = numpy.array(value).view(HashableNumpyArray)
        except ExpressionVariableMissingException:
            raise RuntimeError("Expressions passed into ConstantParameter may not have free variables.")
        
    def get_value(self) -> Union[Real, numpy.ndarray]:
        return self._value

    def __hash__(self) -> int:
        return hash(self._value)

    @property
    def requires_stop(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "<ConstantParameter {0}>".format(self._value)


class MappedParameter(Parameter):
    """A pulse parameter whose value is derived from other parameters via some mathematical
    expression.

    This class bundles an expression with some concrete Parameters which in turn can be derived from other Parameters.

    The dependencies of a MappedParameter instance are defined by the free variables appearing
    in the expression that defines how its value is derived.

    MappedParameter holds a dictionary which assign Parameter objects to these dependencies.
    Evaluation of the MappedParameter will raise a ParameterNotProvidedException if a Parameter
    object is missing for some dependency.
    """

    def __init__(self,
                 expression: Expression,
                 namespace: Optional[Mapping[str, Parameter]]=None) -> None:
        """Create a MappedParameter instance.

        Args:
            expression (Expression): The expression defining how the the value of this
                MappedParameter instance is derived from its dependencies.
             dependencies (Dict(str -> Parameter)): Parameter objects of the dependencies. The objects them selves must
             not change but the parameters might return different values.
        """
        warnings.warn("MappedParameter is deprecated. There should be no interface depending on it", DeprecationWarning,
                      stacklevel=2)

        super().__init__()
        self._expression = expression
        self._namespace = dict() if namespace is None else namespace
        self._cached_value = None

    def _collect_dependencies(self) -> Dict[str, float]:
        # filter only real dependencies from the dependencies dictionary
        try:
            return {parameter_name: self._namespace[parameter_name].get_value()
                    for parameter_name in self._expression.variables}
        except KeyError as key_error:
            raise ParameterNotProvidedException(str(key_error)) from key_error

    def get_value(self) -> Union[Real, numpy.ndarray]:
        """Does not check explicitly if a parameter requires to stop."""
        if self._cached_value is None:
            self._cached_value = self._expression.evaluate_numeric(**self._collect_dependencies())
        return self._cached_value

    @property
    def expression(self):
        return self._expression

    def update_constants(self, new_values: Mapping[str, ConstantParameter]):
        """This is stupid"""
        for parameter_name, parameter in self._namespace.items():
            if hasattr(parameter, '__hash__'):
                # very stupid
                # a constant parameter has a hash function
                if parameter_name in new_values:
                    self._namespace[parameter_name] = new_values[parameter_name]
            else:
                parameter.update_constants(new_values)
        self._cached_value = None

    @property
    def requires_stop(self) -> bool:
        """Does not explicitly check that all parameters are provided if one requires stopping"""
        try:
            return any(self._namespace[v].requires_stop
                       for v in self._expression.variables)
        except KeyError as err:
            raise ParameterNotProvidedException(err.args[0]) from err

    def __eq__(self, other):
        if type(other) == type(self):
            return (self._expression == other._expression and
                    self._namespace == other._namespace)
        else:
            return NotImplemented

    def __repr__(self) -> str:
        try:
            value = self.get_value()
        except ParameterNotProvidedException:
            value = 'nothing'

        return "<MappedParameter {0} evaluating to {1}>".format(
            self._expression, value
        )


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


class ParameterConstrainer:
    """A class that implements the testing of parameter constraints. It is used by the subclassing pulse templates."""
    def __init__(self, *,
                 parameter_constraints: Optional[Iterable[Union[str, ParameterConstraint]]]) -> None:
        if parameter_constraints is None:
            self._parameter_constraints = []
        else:
            self._parameter_constraints = [constraint if isinstance(constraint, ParameterConstraint)
                                           else ParameterConstraint(constraint)
                                           for constraint in parameter_constraints]

    @property
    def parameter_constraints(self) -> List[ParameterConstraint]:
        return self._parameter_constraints

    def validate_parameter_constraints(self, parameters: [str, Union[Parameter, Real]], volatile: Set[str]) -> None:
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
            constraint_parameters = {k: v.get_value() if isinstance(v, Parameter) else v for k, v in parameters.items()}
            if not constraint.is_fulfilled(constraint_parameters, volatile=volatile):
                raise ParameterConstraintViolation(constraint, constraint_parameters)

    def validate_scope(self, scope: Scope):
        volatile = scope.get_volatile_parameters().keys()

        for constraint in self._parameter_constraints:
            if not constraint.is_fulfilled(scope, volatile=volatile):
                constrained_parameters = {parameter_name: scope[parameter_name]
                                          for parameter_name in constraint.affected_parameters}
                raise ParameterConstraintViolation(constraint, constrained_parameters)

    @property
    def constrained_parameters(self) -> Set[str]:
        if self._parameter_constraints:
            return set.union(*(c.affected_parameters for c in self._parameter_constraints))
        else:
            return set()


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
