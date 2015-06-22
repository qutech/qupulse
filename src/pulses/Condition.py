"""STANDARD LIBRARY IMPORTS"""
import logging
from abc import ABCMeta, abstractmethod
import operator

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from pulses.Parameter import Parameter

logger = logging.getLogger(__name__)

class Condition(metaclass = ABCMeta):
    """!@brief A condition on which the execution of a pulse may depend.
    
    Conditions are used for branching and looping of pulses and
    thus relevant for BranchPulseTemplate and LoopPulseTemplate.
    Implementations of Condition may rely on software variables,
    measured data or be mere placeholders for hardware triggers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    @abstractmethod
    def evaluate(self) -> bool:
        pass

class ComparisonCondition(Condition):
    """docstring for ComparisonCondition"""
    # lhs := left hand side of the operation
    # rhs := right hand side of the operation
    def __init__(self, lhs, operator, rhs):
        super().__init__()
        self.lhs = lhs
        self.operator = operator
        self.rhs = rhs
        self._evaluation_function = self.__get_evaluation_function()

    def evaluate(self) -> bool:
        self.lhs_value = None
        if isinstance(self.lhs,Parameter):
            self.lhs_value = self.lhs.get_value
        else:
            #TODO: Figure out how to handle other Types
            pass
        self.rhs_value = None
        if isinstance(self.rhs,Parameter):
            self.rhs_value = self.rhs.get_value
        else:
            #TODO: Figure out how to handle other Types
            pass
        return self._evaluation_function()
        
    def __get_evaluation_function(self):
        if self.operator in ("le","<=","=<"):
            return operator.le
        elif self.operator in ("lt","<"):
            return operator.lt
        elif self.operator in ("eq","=","=="):
            return operator.eq
        elif self.operator in ("ge",">=","=>"):
            return operator.ge
        elif self.operator in ("gt",">"):
            return operator.gt
        elif self.operator in ("ne","!=","=!=","<>"):
            return operator.ne