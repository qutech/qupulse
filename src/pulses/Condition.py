"""STANDARD LIBRARY IMPORTS"""
from logging import getLogger, Logger
from abc import ABCMeta, abstractmethod
import operator

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from Parameter import Parameter

class Condition(metaclass = ABCMeta):
    """docstring for Condition"""
    def __init__(self, *args, **kwargs):
        super(Condition, self).__init__()
        
    @abstractmethod
    def evaluate(self) -> bool:
        pass

class ComparisonCondition(Condition):
    """docstring for ComparisonCondition"""
    # lhs := left hand side of the operation
    # rhs := right hand side of the operation
    def __init__(self, lhs, operator, rhs):
        super(ComparisonCondition, self).__init__()
        super(ComparisonCondition, self).register(self)
        self.lhs = lhs
        self.operator = operator
        self.rhs = rhs
        self.__evaluation_function = self.__get_evaluation_function()

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
        return self.__evaluation_function()
        
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