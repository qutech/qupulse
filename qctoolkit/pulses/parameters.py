"""This module defines parameters and parameter declaration for the usage in pulse modelling.

Classes:
    - Parameter: A base class representing a single pulse parameter.
    - ConstantParameter: A single parameter with a constant value.
    - MappedParameter: A parameter whose value is mathematically computed from another parameter.
    - ParameterDeclaration: The declaration of a parameter within a pulse template.
    - ParameterNotProvidedException.
    - ParameterValueIllegalException.
"""

from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Optional, Union, Dict, Any, Iterable, Set, List
from numbers import Real

import sympy

from qctoolkit.serialization import Serializable, Serializer, ExtendedJSONEncoder
from qctoolkit.expressions import Expression
from qctoolkit.comparable import Comparable

__all__ = ["make_parameter", "ParameterDict", "Parameter", "ConstantParameter",
           "ParameterNotProvidedException", "ParameterConstraintViolation"]


class Parameter(Serializable, Comparable, metaclass=ABCMeta):
    """A parameter for pulses.
    
    Parameter specifies a concrete value which is inserted instead
    of the parameter declaration reference in a PulseTemplate if it satisfies
    the minimum and maximum boundary of the corresponding ParameterDeclaration.
    Implementations of Parameter may provide a single constant value or
    obtain values by computation (e.g. from measurement results).
    """
    def __init__(self) -> None:
        super().__init__(None)

    @abstractmethod
    def get_value(self) -> Real:
        """Compute and return the parameter value."""

    @abstractproperty
    def requires_stop(self) -> bool:
        """Query whether the evaluation of this Parameter instance requires an interruption in
        execution/sequencing, e.g., because it depends on data that is only measured in during the
        next execution.

        Returns:
            True, if evaluating this Parameter instance requires an interruption.
        """

    @property
    def compare_key(self) -> Any:
        return float(self.get_value())
        
        
class ConstantParameter(Parameter):
    """A pulse parameter with a constant value."""
    
    def __init__(self, value: Real) -> None:
        """Create a ConstantParameter instance.

        Args:
            value (Real): The value of the parameter
        """
        super().__init__()
        self.__value = value
        
    def get_value(self) -> Real:
        return self.__value
        
    @property
    def requires_stop(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "<ConstantParameter {0}>".format(self.__value)

    def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
        return dict(type=serializer.get_type_identifier(self), constant=self.__value)

    @staticmethod
    def deserialize(serializer: Serializer, constant: Real) -> 'ConstantParameter':
        return ConstantParameter(constant)


class MappedParameter(Parameter):
    """A pulse parameter whose value is derived from other parameters via some mathematical
    expression.

    The dependencies of a MappedParameter instance are defined by the free variables appearing
    in the expression that defines how its value is derived.

    MappedParameter holds a dictionary which assign Parameter objects to these dependencies.
    Evaluation of the MappedParameter will raise a ParameterNotProvidedException if a Parameter
    object is missing for some dependency.
    """

    def __init__(self,
                 expression: Expression,
                 dependencies: Optional[Dict[str, Parameter]]=None) -> None:
        """Create a MappedParameter instance.

        Args:
            expression (Expression): The expression defining how the the value of this
                MappedParameter instance is derived from its dependencies.
             dependencies (Dict(str -> Parameter)): Parameter objects of the dependencies. May also
                be defined via the dependencies public property. (Optional)
        """
        super().__init__()
        self.__expression = expression
        self.dependencies = dependencies
        if self.dependencies is None:
            self.dependencies = dict()

    def __collect_dependencies(self) -> Iterable[Parameter]:
        # filter only real dependencies from the dependencies dictionary
        try:
            return {dependency_name: self.dependencies[dependency_name]
                    for dependency_name in self.__expression.variables}
        except KeyError as key_error:
            raise ParameterNotProvidedException(str(key_error)) from key_error

    def get_value(self) -> Real:
        if self.requires_stop:
            raise Exception("Cannot evaluate MappedParameter because at least one dependency "
                            "cannot be evaluated.")
        dependencies = self.__collect_dependencies()
        variables = {k: dependencies[k].get_value() for k in dependencies}
        return self.__expression.evaluate_numeric(**variables)

    @property
    def requires_stop(self) -> bool:
        try:
            return any([p.requires_stop for p in self.__collect_dependencies().values()])
        except:
            raise

    def __repr__(self) -> str:
        return "<MappedParameter {0} depending on {1}>".format(
            self.__expression, self.dependencies
        )

    def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
        return dict(type=serializer.get_type_identifier(self),
                    expression=serializer.dictify(self.__expression))

    @staticmethod
    def deserialize(serializer: Serializer, expression: str) -> 'MappedParameter':
        return MappedParameter(serializer.deserialize(expression))


class ParameterConstraint(Comparable):
    def __init__(self, relation: Union[str, sympy.Expr]):
        super().__init__()
        if isinstance(relation, str) and '==' in relation:
            # The '==' operator is interpreted by sympy as exactly, however we need a symbolical evaluation
            self._expression = sympy.Eq(*sympy.sympify(relation.split('==')))
        else:
            self._expression = sympy.sympify(relation)
        if not isinstance(self._expression, sympy.boolalg.Boolean):
            raise ValueError('Constraint is not boolean')

    @property
    def affected_parameters(self) -> Set[str]:
        return set(str(v) for v in self._expression.free_symbols)

    def is_fulfilled(self, parameter: Dict[str, Any]) -> bool:
        if not self.affected_parameters <= set(parameter.keys()):
            raise ParameterNotProvidedException((self.affected_parameters-set(parameter.keys())).pop())
        return bool(self._expression.subs(parameter))

    @property
    def sympified_expression(self) -> sympy.Expr:
        return self._expression

    @property
    def compare_key(self) -> sympy.Expr:
        return self._expression

    def __str__(self) -> str:
        if isinstance(self._expression, sympy.Eq):
            return '{}=={}'.format(self._expression.lhs, self._expression.rhs)
        else:
            return str(self._expression)
ExtendedJSONEncoder.str_constructable_types.add(ParameterConstraint)


class ParameterConstrainer:
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

    def validate_parameter_constraints(self, parameters: [str, Union[Parameter, Real]]) -> None:
        for constraint in self._parameter_constraints:
            constraint_parameters = {k: v.get_value() if isinstance(v, Parameter) else v for k, v in parameters.items()}
            if not constraint.is_fulfilled(constraint_parameters):
                raise ParameterConstraintViolation(constraint, constraint_parameters)

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


class ParameterNotProvidedException(Exception):
    """Indicates that a required parameter value was not provided."""
    
    def __init__(self, parameter_name: str) -> None:
        super().__init__()
        self.parameter_name = parameter_name
        
    def __str__(self) -> str:
        return "No value was provided for parameter '{0}' " \
               "and no default value was specified.".format(self.parameter_name)


class InvalidParameterNameException(Exception):
    def __init__(self, parameter_name: str):
        self.parameter_name = parameter_name

    def __str__(self) -> str:
        return '{} is an invalid parameter name'.format(self.parameter_name)
