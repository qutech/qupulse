"""STANDARD LIBRARY IMPORTS"""
from logging import getLogger, Logger
from abc import ABCMeta, abstractmethod
from typing import Dict

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from Parameter import ParameterDeclaration, Parameter
from PulseTemplate import PulseTemplate

__all__ = ["Table"]
        
class TableValue(metaclass = ABCMeta):
    """docstring for TableValue"""
    def __init__(self):
        super(TableValue, self).__init__()

    @abstractmethod
    def get_value(self, *args, **kwargs):
        pass
        
class ConstantTableValue(object):
    """docstring for ConstantTableValue"""
    def __init__(self, value):
        super(ConstantTableValue, self).__init__()
        super(ConstantTableValue, self).register(self)
        self.value = value
    
    def get_value(self, *args):
        return self.value

class ParameterTableValue(object):
    """docstring for ParameterTableValue"""
    def __init__(self, parameter_name):
        super(ParameterTableValue, self).__init__()
        super(ParameterTableValue, self).register(self)
        self.parameter_name = parameter_name

    def get_value(self, mapping: Dict[str, Parameter]) -> Parameter:
        return mapping[self.parameter_name]
        
class TableEntry(object):
    """docstring for TableEntry"""
    def __init__(self, time: TableValue, voltage: TableValue):
        super(TableEntry, self).__init__()
        self.time = time
        self.voltage = voltage

class Table(PulseTemplate):
    """docstring for Table"""
    def __init__(self):
        super(Table, self).__init__()
        self.entries = {}
        self.parameterDeclarations = dict()

    def add_entry(self, table_entry: TableEntry) -> None:
        self.entries.add(table_entry)
        
    def declare_parameter(self, name: str, **kwargs) -> None:
        self.parameterDeclarations[name] = ParameterDeclaration(kwargs)
