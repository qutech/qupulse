"""STANDARD LIBRARY IMPORTS"""
from abc import ABCMeta, abstractmethod, abstractproperty
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
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_value(self) -> float:
        pass

    @abstractproperty
    def requires_stop(self) -> bool:
        pass
        
        
class ConstantParameter(Parameter):
    """!@brief A pulse parameter with a constant value."""
    
    def __init__(self, value: float) -> None:
        super().__init__()
        self.__value = value
        
    def get_value(self) -> float:
        return self.__value
        
    @property
    def requires_stop(self) -> bool:
        return False
      

class ParameterDeclaration(object):
    """!@brief A declaration of a parameter required by a pulse template.
    
    PulseTemplates may declare parameters to allow for variations of values in an otherwise
    static pulse structure. ParameterDeclaration represents a declaration of such a parameter
    and allows for the definition of boundaries and a default value for a parameter.
    """
    
    def __init__(self, **kwargs) -> None:
        """!@brief Creates a ParameterDeclaration object.
        
        Keyword Arguments:
        @param min: float -- An optional real number specifying the minimum value allowed for the .
        @param max: float -- An optional real number specifying the maximum value allowed.
        @param default: float -- An optional real number specifying a default value for the declared pulse template parameter.
        """
        super().__init__()
        self._min_value = None
        self._max_value = None
        self._default_value = None
        for key in kwargs:
            value = kwargs[key]
            if isinstance(value, numbers.Real):
                if key == "min":
                    self._min_value = value
                elif key == "max":
                    self._max_value = value
                elif key == "default":
                    self._default_value = value
                else:
                    raise ValueError("{0} is not a valid argument.".format(key))
            else:
                raise TypeError("Argument {0}={1} must be of type float.".format(key, value))
        if self._min_value is not None:
            if (self._max_value is not None) and (self._min_value > self._max_value):
                raise ValueError("Max value ({0}) is less than min value ({1}).".format(self._max_value, self._min_value))
            if (self._default_value is not None) and (self._min_value > self._default_value):
                raise ValueError("Default value({0}) is less than min value ({1}).".format(self._default_value, self._min_value))
        if (self._max_value is not None) and (self._default_value is not None) and (self._default_value > self._max_value):
            raise ValueError("Default value ({0}) is greater than max value ({1}).".format(self._default_value, self._max_value))
        
    
    def get_min_value(self) -> Optional[float]:
        """!@brief Return this ParameterDeclaration's minimum value."""
        return self._min_value
    
    def get_max_value(self) -> Optional[float]:
        """!@brief Return this ParameterDeclaration's maximum value."""
        return self._max_value
        
    def get_default_value(self) -> Optional[float]:
        """!@brief Return this ParameterDeclaration's default value"""
        return self._default_value
        
    def get_default_parameter(self) -> ConstantParameter:
        """!@brief Creates a ConstantParameter object holding the default value of this ParameterDeclaration."""
        if (self._default_value is None):
            raise NoDefaultValueException()
        return ConstantParameter(self._default_value)
    
    min_value = property(get_min_value)
    max_value = property(get_max_value)
    default_value = property(get_default_value)

    def is_parameter_valid(self, p: Parameter) -> bool:
        """!@brief Checks whether a given parameter satisfies this ParameterDeclaration.
        
        A parameter is valid if the following two statements hold:
        - If the declaration specifies a minimum value, the parameter's value must be greater or equal
        - If the declaration specifies a maximum value, the parameter's value must be less or equal
        """
        is_valid = True
        is_valid &= (self._min_value is None or self._min_value <= p.get_value())
        is_valid &= (self._max_value is None or self._max_value >= p.get_value())
        return is_valid
        
class TimeParameterDeclaration(ParameterDeclaration):
    """!@brief A TimeParameterDeclaration declares a parameter that is used as a time value.
    
    All values must be natural numbers.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        if self._min_value is None:
            self._min_value = 0
        
        for key in kwargs:
            value = kwargs[key]
            if not isinstance(value, numbers.Integral):
                raise TypeError("{0} value {1} is not an integer.".format(key, value))
            if value < 0:
                raise ValueError("{0} value {1} is less than zero.".format(key, value))
        
    def is_parameter_valid(self, p: Parameter) -> bool:
        if not isinstance(p.get_value(), numbers.Integral):
            return False
        return super().is_parameter_valid(p)
        
class NoDefaultValueException(Exception):
    """!@brief Indicates that a ParameterDeclaration specifies no default value."""
    def __init__(self) -> None:
        super().__init__()
        
    def __str__(self) -> str:
        return "A default value was not specified in this ParameterDeclaration."
