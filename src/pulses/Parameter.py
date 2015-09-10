"""STANDARD LIBRARY IMPORTS"""
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Optional, Union, Dict, Tuple
import numbers
import logging

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from .Serializer import Serializable

logger = logging.getLogger(__name__)

class Parameter(metaclass = ABCMeta):
    """A parameter for pulses.
    
    Parameter specifies a concrete value which is inserted instead
    of the parameter declaration reference in a PulseTemplate if it satisfies
    the minimum and maximum boundary of the corresponding ParameterDeclaration.
    Implementations of Parameter may provide a single constant value or
    obtain values by computation (e.g. from measurement results).
    """
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_value(self) -> float:
        pass

    @abstractproperty
    def requires_stop(self) -> bool:
        pass
    
    def __float__(self) -> float:
        return float(self.get_value())
        
        
class ConstantParameter(Parameter):
    """A pulse parameter with a constant value."""
    
    def __init__(self, value: float) -> None:
        super().__init__()
        self.__value = value
        
    def get_value(self) -> float:
        return self.__value
        
    @property
    def requires_stop(self) -> bool:
        return False

    def __repr__(self):
        return "<ConstantParameter {0}>".format(self.__value)

    
ConstantParameter.register(numbers.Real)
      
class ParameterValueProvider(metaclass = ABCMeta):
    
    @abstractmethod
    def get_value(self, parameters: Dict[str, Parameter]) -> float:
        pass
      
class ParameterDeclaration(Serializable, ParameterValueProvider):
    """A declaration of a parameter required by a pulse template.
    
    PulseTemplates may declare parameters to allow for variations of values in an otherwise
    static pulse structure. ParameterDeclaration represents a declaration of such a parameter
    and allows for the definition of boundaries and a default value for a parameter.
    """
    
    BoundaryValue = Union[float, 'ParameterDeclaration']
    
    def __init__(self, name: str, min: BoundaryValue = float('-inf'), max: BoundaryValue = float('+inf'), default: Optional[float] = None) -> None:
        """Creates a ParameterDeclaration object.
        
        Args:
            min (float, ParameterDeclaration): An optional real number or ParameterDeclaration object specifying the minimum value allowed.
            max (float, ParameterDeclaration): An optional real number or ParameterDeclaration object specifying the maximum value allowed.
            default (float): An optional real number specifying a default value for the declared pulse template parameter.
        """
        super().__init__(name)
        self.__min_value = float('-inf')
        self.__max_value = float('+inf')
        self.__default_value = default # type: Optional[float]
        self.min_value = min # type: BoundaryValue
        self.max_value = max # type: BoundaryValue
            
        self.__assert_values_valid()

    def __assert_values_valid(self) -> None:
        if self.absolute_min_value > self.absolute_max_value:
            raise ValueError("Max value ({0}) is less than min value ({1}).".format(self.max_value, self.min_value))
        
        if isinstance(self.min_value, ParameterDeclaration):
            if self.min_value.absolute_max_value > self.absolute_max_value:
                raise ValueError("Max value ({0}) is less than min value ({1}).".format(self.max_value, self.min_value))
            
        if isinstance(self.max_value, ParameterDeclaration):
            if self.max_value.absolute_min_value < self.absolute_min_value:
                raise ValueError("Max value ({0}) is less than min value ({1}).".format(self.max_value, self.min_value))
            
        if self.default_value is not None and self.absolute_min_value > self.default_value:
            raise ValueError("Default value ({0}) is less than min value ({1}).".format(self.default_value, self.min_value))
        
        if self.default_value is not None and self.absolute_max_value < self.default_value:
            raise ValueError("Default value ({0}) is greater than max value ({1}).".format(self.__default_value, self.__max_value))
        
    @property
    def name(self) -> str:
        return self.identifier
        
    @property
    def min_value(self) -> BoundaryValue:
        """Return this ParameterDeclaration's minimum value or reference."""
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
        """Return this ParameterDeclaration's maximum value or reference."""
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
        """Return this ParameterDeclaration's default value."""
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
        """Checks whether a given parameter satisfies this ParameterDeclaration.
        
        A parameter is valid if all of the following statements hold:
        - If the declaration specifies a minimum value, the parameter's value must be greater or equal
        - If the declaration specifies a maximum value, the parameter's value must be less or equal
        """
        parameter_value = float(p)
        is_valid = True
        is_valid &= self.absolute_min_value <= parameter_value
        is_valid &= self.absolute_max_value >= parameter_value
        return is_valid
    
    def check_parameter_set_valid(self, parameters: Dict[str, Parameter]) -> bool:
        parameter_value = self.get_value(parameters)
        
        # get actual instantiated values for boundaries.
        min_value = self.min_value
        if isinstance(min_value, ParameterDeclaration):
            min_value = min_value.get_value(parameters)
            
        max_value = self.max_value
        if isinstance(max_value, ParameterDeclaration):
            max_value = max_value.get_value(parameters)
            
        return min_value <= parameter_value and max_value >= parameter_value
    
    def get_value(self, parameters: Dict[str, Parameter]) -> float:
        try:
            return float(parameters[self.name]) # float() wraps get_value for Parameters and works for normal floats also
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
        return "{4} '{0}', range ({1}, {2}), default {3}".format(self.name, min_value_str, max_value_str, self.default_value, type(self))

    def __repr__(self) -> str:
        return "<"+self.__str__()+">"

    def __compute_compare_key(self) -> Tuple[str, Union[float, str], Union[float, str], Optional[float]]:
        min_value = self.min_value
        if isinstance(min_value, ParameterDeclaration):
            min_value = min_value.name
        max_value = self.max_value
        if isinstance(max_value, ParameterDeclaration):
            max_value = max_value.name
        return (self.name, min_value, max_value, self.default_value)

    def __eq__(self, other) -> bool:
        return (isinstance(other, ParameterDeclaration) and
                self.__compute_compare_key() == other.__compute_compare_key())

    def __hash__(self) -> int:
        return hash(self.__compute_compare_key())
        
# class ImmutableParameterDeclaration(ParameterDeclaration):
#
#     def __init__(self, parameter_declaration: ParameterDeclaration) -> None:
#         self.__parameter_declaration = parameter_declaration
#         super().__init__(parameter_declaration.name)
#
#     @property
#     def name(self) -> str:
#         return self.__parameter_declaration.name
#
#     @property
#     def min_value(self) -> ParameterDeclaration.BoundaryValue:
#         """Return this ParameterDeclaration's minimum value or reference."""
#         min_value = self.__parameter_declaration.min_value
#         if isinstance(min_value, ParameterDeclaration):
#             min_value = ImmutableParameterDeclaration(min_value)
#         return min_value
#
#     @min_value.setter
#     def min_value(self, value: ParameterDeclaration.BoundaryValue) -> None:
#         """Set this ParameterDeclaration's minimum value or reference."""
#         raise Exception("An ImmutableParameterDeclaration may not be changed.")
#
#     @property
#     def max_value(self) ->  ParameterDeclaration.BoundaryValue:
#         """Return this ParameterDeclaration's maximum value or reference."""
#         max_value = self.__parameter_declaration.max_value
#         if isinstance(max_value, ParameterDeclaration):
#             max_value = ImmutableParameterDeclaration(max_value)
#         return max_value
#
#     @max_value.setter
#     def max_value(self, value: ParameterDeclaration.BoundaryValue) -> None:
#         """Set this ParameterDeclaration's maximum value or reference."""
#         raise Exception("An ImmutableParameterDeclaration may not be changed.")
#
#     @property
#     def default_value(self) -> Optional[float]:
#         """Return this ParameterDeclaration's default value."""
#         return self.__parameter_declaration.default_value
        
        
class ParameterNotProvidedException(Exception):
    """Indicates that a required parameter value was not provided."""
    
    def __init__(self, parameter_name: str) -> None:
        super().__init__()
        self.parameter_name = parameter_name
        
    def __str__(self) -> str:
        return "No value was provided for parameter {0} and no default value was specified.".format(self.parameter_name)
