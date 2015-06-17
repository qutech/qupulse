"""STANDARD LIBRARY IMPORTS"""
from abc import ABCMeta, abstractmethod
from typing import Optional
import numbers
import logging

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""    

logger = logging.getLogger(__name__)

class Parameter(metaclass = ABCMeta):
    """!@brief A parameter for pulses.
    
    Parameter specifies a concrete value which is inserted instead
    of the parameter declaration reference in a PulseTemplate if it satisfies
    the minimum and maximum boundary of the corresponding ParameterDeclaration.
    Implementations of Parameter may provide a single constant value or
    obtain values by computation (e.g. from measurement results).
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_value(self) -> float:
        pass

    @abstractmethod
    def requires_stop(self) -> bool:
        pass
        
        
class ConstantParameter(Parameter):
    """!@brief A pulse parameter with a constant value."""
    
    def __init__(self, value: float):
        super().__init__()
        self._value = value
        
    def get_value(self) -> float:
        return self._value
        
    def requires_stop(self) -> bool:
        return False
      

class ParameterDeclaration(object):
    """!@brief A declaration of a parameter required by a pulse template.
    
    PulseTemplates may declare parameters to allow for variations of values in an otherwise
    static pulse structure. ParameterDeclaration represents a declaration of such a parameter
    and allows for the definition of boundaries and a default value for a parameter.
    """
    
    def __init__(self, **kwargs):
        """!@brief Creates a ParameterDeclaration object.
        
        Keyword Arguments:
        @param min: float -- An optional real number specifying the minimum value allowed for the .
        @param max: float -- An optional real number specifying the maximum value allowed.
        @param default: float -- An optional real number specifying a default value for the declared pulse template parameter.
        """
        super().__init__()
        self._minValue = None
        self._maxValue = None
        self._defaultValue = None
        for key in kwargs:
            value = kwargs[key]
            if isinstance(value, numbers.Real):
                if key == "min":
                    self._minValue = value
                elif key == "max":
                    self._maxValue = value
                elif key == "default":
                    self._defaultValue = value
                else:
                    raise ValueError("{0} is not a valid argument.".format(key))
            else:
                raise ValueError("{0}={1} is not a valid argument.".format(key, value))
        if self._minValue is not None:
            if (self._maxValue is not None) and (self._minValue > self._maxValue):
                raise ValueError("Max value ({0}) is less than min value ({1}).".format(self._maxValue, self._minValue))
            if (self._defaultValue is not None) and (self._minValue > self._defaultValue):
                raise ValueError("Default value({0}) is less than min value ({1}).".format(self._defaultValue, self._minValue))
        if (self._maxValue is not None) and (self._defaultValue is not None) and (self._defaultValue > self._maxValue):
            raise ValueError("Default value ({0}) is greater than max value ({1}).".format(self._defaultValue, self._maxValue))
                
    
    def get_min_value(self) -> Optional[float]:
        """!@brief Return this ParameterDeclaration's minimum value."""
        return self._minValue
    
    def get_max_value(self) -> Optional[float]:
        """!@brief Return this ParameterDeclaration's maximum value."""
        return self._maxValue
        
    def get_default_value(self) -> Optional[float]:
        """!@brief Return this ParameterDeclaration's default value"""
        return self._defaultValue
        
    def get_default_parameter(self) -> ConstantParameter:
        """!@brief Creates a ConstantParameter object holding the default value of this ParameterDeclaration."""
        if (self._defaultValue is None):
            raise NoDefaultValueException()
        return ConstantParameter(self._defaultValue)
    
    minValue = property(get_min_value)
    maxValue = property(get_max_value)
    defaultValue = property(get_default_value)

    def is_parameter_valid(self, p: Parameter) -> bool:
        """!@brief Checks whether a given parameter satisfies this ParameterDeclaration.
        
        A parameter is valid if the following two statements hold:
        - If the declaration specifies a minimum value, the parameter's value must be greater or equal
        - If the declaration specifies a maximum value, the parameter's value must be less or equal
        """
        isValid = True
        isValid &= (self._minValue is None or self._minValue <= p.get_value())
        isValid &= (self._maxValue is None or self._maxValue >= p.get_value())
        return isValid
        
class NoDefaultValueException(Exception):
    """!@brief Indicates that a ParameterDeclaration specifies no default value."""
    def __init__(self):
        super().__init__()
        
    def __str__(self) -> str:
        return "A default value was not specified in this ParameterDeclaration."
