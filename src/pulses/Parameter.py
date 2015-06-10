"""STANDARD LIBRARY IMPORTS"""
from abc import ABCMeta, abstractmethod
from typing import Optional
from numbers import Real

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""    

class Parameter(metaclass = ABCMeta):
    """A parameter for pulses."""
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_value(self) -> Real:
        pass

    @abstractmethod
    def requires_stop(self) -> bool:
        pass
        
        
class ConstantParameter(Parameter):
    """A parameter with a constant value."""
    
    def __init__(self, value: Real):
        super().__init__()
        self.register(self)
        self.__value = value
        if (self.__value is None):
            self.value = 0
        
    def get_value(self) -> Real:
        return self.__value
        
    def requires_stop(self) -> bool:
        return False
      

class ParameterDeclaration(object):
    """A declaration of a parameter required by a pulse template.
    
    PulseTemplates may declare parameters to allow for variations of values in an otherwise
    static pulse structure. ParameterDeclaration represents a declaration of such a parameter
    and allows for the definition of boundaries and a default value for a parameter.
    """
    
    def __init__(self, **kwargs):
        """Creates a ParameterDeclaration object.
        
        Keyword Arguments:
        min or minValue -- A real number specifying the minimum value allowed for the .
        max or maxValue -- A real number specifying the maximum value allowed.
        default or defaultValue -- A real number specifying a default value for the declared pulse template parameter.
        """
        super().__init__()
        self.__minValue = None
        self.__maxValue = None
        self.__defaultValue = None
        for key in kwargs:
            value = kwargs[key]
            if isinstance(value, Real):
                if (key == "min" or key == "minValue"):
                    self.__minValue == value
                elif (key == "max" or key == "maxValue"):
                    self.__maxValue == value
                elif (key == "default" or key == "defaultValue"):
                    self.__defaultValue = value
        
    def get_min_value(self) -> Optional[Real]:
        return self.__minValue
    
    def get_max_value(self) -> Optional[Real]:
        return self.__maxValue
        
    def get_default_value(self) -> Optional[Real]:
        return self.__defaultValue
        
    def get_default_parameter(self) -> ConstantParameter:
        """Creates a ConstantParameter object holding the default value of this ParameterDeclaration."""
        if (self.__defaultValue is None):
            raise NoDefaultValueException()
        return ConstantParameter(self.__defaultValue)
    
    minValue = property(get_min_value)
    maxValue = property(get_max_value)
    defaultValue = property(get_default_value)

    def is_parameter_valid(self, p: Parameter) -> bool:
        """Checks whether a given parameter satisfies this ParameterDeclaration.
        
        A parameter is valid if the following two statements hold:
        - If the declaration specifies a minimum value, the parameter's value must be greater or equal
        - If the declaration specifies a maximum value, the parameter's value must be less or equal
        """
        isValid = True
        isValid &= (self.__minValue is None or self.__minValue <= p.get_value())
        isValid &= (self.__maxValue is None or self.__maxValue >= p.get_value())
        return isValid
        
class NoDefaultValueException(Exception):
    """Indicates that a ParameterDeclaration specifies no default value."""
    def __init__(self):
        super().__init__()
        
    def __str__(self) -> str:
        return "A default value was not specified in this ParameterDeclaration."