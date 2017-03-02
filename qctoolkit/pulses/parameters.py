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
from typing import Optional, Union, Dict, Any, Iterable

from qctoolkit.serialization import Serializable, Serializer
from qctoolkit.expressions import Expression
from qctoolkit.comparable import Comparable

__all__ = ["make_parameter", "ParameterDict", "Parameter", "ParameterDeclaration", "ConstantParameter",
           "ParameterNotProvidedException", "ParameterValueIllegalException"]


def make_parameter(value):
    """Convenience function """
    if isinstance(value, Parameter):
        return value
    if isinstance(value, Number):
        return ConstantParameter(value)
    if isinstance(value, str):
        return MappedParameter(Expression(value))
    raise TypeError('Can not convert object of type {} to a parameter'.format(type(value)))


class ParameterDict(dict):
    """Conve"""
    def __init__(self, *args, **kwargs):
        super().__init__(
            *((k, make_parameter(v)) for k, v in args),
            **dict((k, make_parameter(v)) for k, v in kwargs.items())
        )

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, make_parameter(value))


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
    def get_value(self) -> float:
        """Compute and return the parameter value."""

    @abstractproperty
    def requires_stop(self) -> bool:
        """Query whether the evaluation of this Parameter instance requires an interruption in
        execution/sequencing, e.g., because it depends on data that is only measured in during the
        next execution.

        Returns:
            True, if evaluating this Parameter instance requires an interruption.
        """
    
    def __float__(self) -> float:
        return float(self.get_value())

    @property
    def compare_key(self) -> Any:
        return float(self.get_value())
        
        
class ConstantParameter(Parameter):
    """A pulse parameter with a constant value."""
    
    def __init__(self, value: float) -> None:
        """Create a ConstantParameter instance.

        Args:
            value (float): The value of the parameter
        """
        super().__init__()
        self.__value = value
        
    def get_value(self) -> float:
        return self.__value
        
    @property
    def requires_stop(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "<ConstantParameter {0}>".format(self.__value)

    def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
        return dict(type=serializer.get_type_identifier(self), constant=self.__value)

    @staticmethod
    def deserialize(serializer: Serializer, constant: float) -> 'ConstantParameter':
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
                    for dependency_name in self.__expression.variables()}
        except KeyError as key_error:
            raise ParameterNotProvidedException(str(key_error)) from key_error

    def get_value(self) -> float:
        if self.requires_stop:
            raise Exception("Cannot evaluate MappedParameter because at least one dependency "
                            "cannot be evaluated.")
        dependencies = self.__collect_dependencies()
        variables = {k: float(dependencies[k]) for k in dependencies}
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


class ParameterDeclaration(Serializable, Comparable):
    """A declaration of a parameter required by a pulse template.
    
    PulseTemplates may declare parameters to allow for variations of values in an otherwise
    static pulse structure. ParameterDeclaration represents a declaration of such a parameter
    and allows for the definition of boundaries and a default value for a parameter. Boundaries
    may be either defined as constant value or as references to another ParameterDeclaration object.
    """
    
    BoundaryValue = Union[float, 'ParameterDeclaration']
    
    def __init__(self, name: str,
                 min: BoundaryValue=float('-inf'),
                 max: BoundaryValue=float('+inf'),
                 default: Optional[float]=None) -> None:
        """Creates a ParameterDeclaration object.
        
        Args:
            name (str): A name for the declared parameter. The name must me a valid variable name.
            min (float or ParameterDeclaration): An optional real number or
                ParameterDeclaration object specifying the minimum value allowed. (default: -inf)
            max (float or ParameterDeclaration): An optional real number or
                ParameterDeclaration object specifying the maximum value allowed. (default: +inf)
            default (float): An optional real number specifying a default value for the declared
                pulse template parameter.
        """
        super().__init__(None)
        if not name.isidentifier():
            raise InvalidParameterNameException(name)

        self.__name = name
        self.__min_value = float('-inf')
        self.__max_value = float('+inf')
        self.__default_value = default # type: Optional[float]
        self.min_value = min # type: BoundaryValue
        self.max_value = max # type: BoundaryValue
            
        self.__assert_values_valid()

    def __assert_values_valid(self) -> None:
        # ensures that min <= default <= max or raises a ValueError
        if self.absolute_min_value > self.absolute_max_value:
            raise ValueError("Max value ({0}) is less than min value ({1}).".format(
                    self.max_value, self.min_value
                )
            )
        
        if isinstance(self.min_value, ParameterDeclaration):
            if self.min_value.absolute_max_value > self.absolute_max_value:
                raise ValueError("Max value ({0}) is less than min value ({1}).".format(
                        self.max_value, self.min_value
                    )
                )
            
        if isinstance(self.max_value, ParameterDeclaration):
            if self.max_value.absolute_min_value < self.absolute_min_value:
                raise ValueError("Max value ({0}) is less than min value ({1}).".format(
                        self.max_value, self.min_value
                    )
                )
            
        if self.default_value is not None and self.absolute_min_value > self.default_value:
            raise ValueError("Default value ({0}) is less than min value ({1}).".format(
                    self.default_value, self.min_value
                )
            )
        
        if self.default_value is not None and self.absolute_max_value < self.default_value:
            raise ValueError("Default value ({0}) is greater than max value ({1}).".format(
                    self.__default_value, self.__max_value
                )
            )
        
    @property
    def name(self) -> str:
        """The name of the declared parameter."""
        return self.__name
        
    @property
    def min_value(self) -> BoundaryValue:
        """This ParameterDeclaration's minimum value or reference."""
        return self.__min_value
    
    @min_value.setter
    def min_value(self, value: BoundaryValue) -> None:
        """Set this ParameterDeclaration's minimum value or reference."""
        old_value = self.__min_value
        self.__min_value = value
        try:
            if (isinstance(value, ParameterDeclaration) and
                    (isinstance(value.max_value, ParameterDeclaration) or
                     value.absolute_max_value == float('+inf'))):
                value.__internal_set_max_value(self)
            self.__assert_values_valid()
        except:
            self.__min_value = old_value
            raise
        
    def __internal_set_min_value(self, value: BoundaryValue) -> None:
        old_value = self.__min_value
        self.__min_value = value
        try:
            self.__assert_values_valid()
        except:
            self.__min_value = old_value
            raise
    
    @property
    def max_value(self) ->  BoundaryValue:
        """This ParameterDeclaration's maximum value or reference."""
        return self.__max_value
    
    @max_value.setter
    def max_value(self, value: BoundaryValue) -> None:
        """Set this ParameterDeclaration's maximum value or reference."""
        old_value = self.__max_value
        self.__max_value = value
        try:
            if (isinstance(value, ParameterDeclaration) and
                    (isinstance(value.min_value, ParameterDeclaration) or
                     value.absolute_min_value == float('-inf'))):
                value.__internal_set_min_value(self)
            self.__assert_values_valid()
        except:
            self.__max_value = old_value
            raise
        
    def __internal_set_max_value(self, value: BoundaryValue) -> None:
        old_value = self.__max_value
        self.__max_value = value
        try:
            self.__assert_values_valid()
        except:
            self.__max_value = old_value
            raise
        
    @property
    def default_value(self) -> Optional[float]:
        """This ParameterDeclaration's default value."""
        return self.__default_value
    
    @property
    def absolute_min_value(self) -> float:
        """Return this ParameterDeclaration's minimum value.
        
        If the minimum value of this ParameterDeclaration instance is a reference to another
        instance, references are resolved until a concrete value or None is obtained.
        """ 
        if isinstance(self.min_value, ParameterDeclaration):
            return self.min_value.absolute_min_value
        else:
            return self.min_value

    @property
    def absolute_max_value(self) -> float:
        """Return this ParameterDeclaration's maximum value.
        
        If the maximum value of this ParameterDeclaration instance is a reference to another
        instance, references are resolved until a concrete value or None is obtained.
        """
        if isinstance(self.max_value, ParameterDeclaration):
            return self.max_value.absolute_max_value
        else:
            return self.max_value

    def is_parameter_valid(self, p: Parameter) -> bool:
        """Check whether a given parameter satisfies this ParameterDeclaration statically.
        
        A parameter is valid if all of the following statements hold:
        - If the declaration specifies a minimum value, the parameter's value must be greater or
            equal
        - If the declaration specifies a maximum value, the parameter's value must be less or equal

        Checks only against the static boundaries. For example, if the min value for this
        ParameterDeclaration would be another ParameterDeclaration named 'foo' with a min value of
        3.5, this method only checks whether the given parameter value is greater than or equal to
        3.5. However, the implicit meaning of the reference minimum declaration is, that the value
        provided for this ParameterDeclaration must indeed by greater than or equal than the value
        provided for the referenced minimum declaration.

        Args:
            p (Parameter): The Parameter object checked for validity.
        Returns:
            True, if p is a valid parameter for this ParameterDeclaration.
        See also:
            check_parameter_set_valid()
        """
        parameter_value = float(p)
        is_valid = True
        is_valid &= self.absolute_min_value <= parameter_value
        is_valid &= self.absolute_max_value >= parameter_value
        return is_valid
    
    def get_value(self, parameters: Dict[str, Parameter]) -> float:
        """Retrieve the value of the parameter corresponding to this ParameterDeclaration object
        from a set of parameter assignments.

        Args:
            parameters (Dict(str -> Parameter)): A mapping of parameter names to Parameter objects.
        Returns:
            The value of the parameter corresponding to this ParameterDeclaration as a float.
        Raises:
            ParameterNotProvidedException if no parameter is assigned to the name of this
                ParameterDeclaration or any other ParameterDeclaration required to evaluate the
                boundary conditions of this ParameterDeclaration.
            ParameterValueIllegalException if a parameter exists but its value exceeds the bounds
                specified by the corresponding ParameterDeclaration.
        """
        value = self.__get_value_internal(parameters)
        if not self.check_parameter_set_valid(parameters):
            raise ParameterValueIllegalException(self, value)
        return value

    def check_parameter_set_valid(self, parameters: Dict[str, Parameter]) -> bool:
        """Check whether an entire set of parameters is consistent with this ParameterDeclaration.

        Recursively evaluates referenced min and max ParameterDeclarations (if existent) and checks
        whether the values provided for these compare correctly, i.e., does not only perform static
        boundary checks.

        Args:
            parameters (Dict(str -> Parameter)): A mapping of parameter names to Parameter objects.
        Returns:
            True, if the values provided for the parameters satisfy all boundary checks for this
                ParameterDeclaration.
        Raises:
            ParameterNotProvidedException if no parameter is assigned to the name of this
                ParameterDeclaration or any other ParameterDeclaration required to evaluate the
                boundary conditions of this ParameterDeclaration.
            ParameterValueIllegalException if a parameter exists but its value exceeds the bounds
                specified by the corresponding ParameterDeclaration.
        """
        parameter_value = self.__get_value_internal(parameters)

        # get actual instantiated values for boundaries.
        min_value = self.min_value
        if isinstance(min_value, ParameterDeclaration):
            min_value = min_value.__get_value_internal(parameters)

        max_value = self.max_value
        if isinstance(max_value, ParameterDeclaration):
            max_value = max_value.__get_value_internal(parameters)

        return min_value <= parameter_value and max_value >= parameter_value

    def __get_value_internal(self, parameters: Dict[str, Parameter]) -> float:
        try:
            return float(parameters[self.name]) # float() wraps get_value for Parameters and works
                                                # for normal floats also
        except KeyError:
            if self.default_value is not None:
                return self.default_value
            else:
                raise ParameterNotProvidedException(self.name)

    def __str__(self) -> str:
        min_value_str = self.absolute_min_value
        if isinstance(self.min_value, ParameterDeclaration):
            min_value_str = "Parameter '{0}' (min {1})".format(self.min_value.name, min_value_str)
        max_value_str = self.absolute_max_value
        if isinstance(self.max_value, ParameterDeclaration):
            max_value_str = "Parameter '{0}' (max {1})".format(self.max_value.name, max_value_str)
        return "{4} '{0}', range ({1}, {2}), default {3}".format(
            self.name, min_value_str, max_value_str, self.default_value, type(self)
        )

    def __repr__(self) -> str:
        return "<"+self.__str__()+">"

    @property
    def compare_key(self) -> Any:
        min_value = self.min_value
        if isinstance(min_value, ParameterDeclaration):
            min_value = min_value.name
        max_value = self.max_value
        if isinstance(max_value, ParameterDeclaration):
            max_value = max_value.name
        return (self.name, min_value, max_value, self.default_value)

    def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
        data = dict()

        min_value = self.min_value
        if isinstance(min_value, ParameterDeclaration):
            min_value = min_value.name

        max_value = self.max_value
        if isinstance(max_value, ParameterDeclaration):
            max_value = max_value.name

        data['name'] = self.name
        data['min_value'] = min_value
        data['max_value'] = max_value
        data['default_value'] = self.default_value
        data['type'] = serializer.get_type_identifier(self)

        return data

    @staticmethod
    def deserialize(serializer: Serializer,
                    name: str,
                    min_value: Union[str, float],
                    max_value: Union[str, float],
                    default_value: float) -> 'ParameterDeclaration':
        if isinstance(min_value, str):
            min_value = float("-inf")
        if isinstance(max_value, str):
            max_value = float("+inf")
        return ParameterDeclaration(name, min=min_value, max=max_value, default=default_value)

        
class ParameterNotProvidedException(Exception):
    """Indicates that a required parameter value was not provided."""
    
    def __init__(self, parameter_name: str) -> None:
        super().__init__()
        self.parameter_name = parameter_name
        
    def __str__(self) -> str:
        return "No value was provided for parameter '{0}' " \
               "and no default value was specified.".format(self.parameter_name)


class ParameterValueIllegalException(Exception):
    """Indicates that the value provided for a parameter is illegal, i.e., is outside the
    parameter's bounds or of wrong type."""

    def __init__(self, parameter_declaration: ParameterDeclaration, parameter_value: float) -> None:
        super().__init__()
        self.parameter_value = parameter_value
        self.parameter_declaration = parameter_declaration

    def __str__(self) -> str:
        return "The value {0} provided for parameter {1} is illegal (min = {2}, max = {3})".format(
            self.parameter_value, self.parameter_declaration.name,
            self.parameter_declaration.min_value, self.parameter_declaration.max_value)


class InvalidParameterNameException(Exception):
    def __init__(self, parameter_name: str):
        self.parameter_name = parameter_name

    def __str__(self):
        return '{} is an invalid parameter name'.format(self.parameter_name)
