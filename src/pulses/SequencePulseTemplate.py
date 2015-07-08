"""STANDARD LIBRARY IMPORTS"""
import logging
from typing import Dict, List, Tuple
from functools import partial
"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from PulseTemplate import PulseTemplate
from Parameter import ParameterDeclaration, TimeParameterDeclaration, Parameter

logger = logging.getLogger(__name__)

class SequencePulseTemplate(PulseTemplate):
    """!@brief A sequence of different PulseTemplates.
    
    SequencePulseTemplate allows to group smaller
    PulseTemplates (subtemplates) into on larger sequence,
    i.e., when instantiating a pulse from a SequencePulseTemplate
    all pulses instantiated from the subtemplates are queued for
    execution right after one another.
    SequencePulseTemplate allows to specify a mapping of
    parameter declarations from its subtemplates, enabling
    renaming and mathematical transformation of parameters.
    The default behavior is to exhibit the union of parameter
    declarations of all subtemplates. If two subpulses declare
    a parameter with the same name, it is mapped to both. If the
    declarations define different minimal and maximal values, the
    more restricitive is chosen, if possible. Otherwise, an error
    is thrown and an explicit mapping must be specified.
    """
    
    def __init__(self):
        super().__init__()
        self.subtemplates = [] # type: List[PulseTemplate]
        self.mapping = Mapping() #type: Mapping
        self.__is_interruptable = None
        
    def get_time_parameter_names(self) -> set[str]:
        time_parameter_names = set()
        for subtemplate in self.subtemplates:
            for time_parameter_name in subtemplate.get_time_parameter_names():
                if time_parameter_name in self.mapping.get_targets():
                    time_parameter_names.add(self.mapping.get_targets()[time_parameter_name])
                else:
                    time_parameter_names.add(time_parameter_name)
        return time_parameter_names
        
    def get_voltage_parameter_names(self) -> set[str]:
        voltage_parameter_names = set()
        for subtemplate in self.subtemplates:
            for voltage_parameter_name in subtemplate.get_time_parameter_names():
                if voltage_parameter_name in self.mapping.get_targets():
                    voltage_parameter_names.add(self.mapping.get_targets()[voltage_parameter_name])
                else:
                    voltage_parameter_names.add(voltage_parameter_name)
        return voltage_parameter_names
    
    def get_time_parameter_declaration(self) -> dict[str, TimeParameterDeclaration]:
        time_parameter_declaration = {}
        for subtemplate in self.subtemplates:
            for parameter,declaration in subtemplate.get_time_parameter_declaration():
                if parameter in self.mapping.get_targets():
                    #Rename the parameter according to the mapping. The target name becomes the source name.
                    time_parameter_declaration[self.mapping.get_targets()[parameter]] = declaration
                else:
                    time_parameter_declaration[parameter] = declaration
        return time_parameter_declaration
           
    def get_voltage_parameter_declarations(self) -> Dict[str, ParameterDeclaration]:
        voltage_parameter_declaration = {}
        for subtemplate in self.subtemplates:
            for parameter,declaration in subtemplate.get_voltage_parameter_declaration():
                if parameter in self.mapping.get_targets():
                    #Rename the parameter according to the mapping. The target name becomes the source name.
                    voltage_parameter_declaration[self.mapping.get_targets()[parameter]] = declaration
                else:
                    voltage_parameter_declaration[parameter] = declaration
          
    def get_measurement_windows(self) -> List[Tuple[float, float]]:
        measuerement_windows = []
        for subtemplate in self.subtemplates:
            measuerement_windows = measuerement_windows + subtemplate.get_measurement_windows()
        return measuerement_windows
    
    def is_interruptable(self) -> bool:
        interruptable = True
        for subtemplate in self.subtemplates:
            interruptable = interruptable and subtemplate.is_interruptable()
        return interruptable

    def get_mapping(self):
        return self.mapping
    
class Mapping(object):
    def __init__(self):
        super().__init__()
        self.functions = {}
        self.__targets = {}
    
    def add_mapping_function(self,source:str,target:str,func:function,*args,**kwargs):
        if source not in self.functions:
            self.functions[source] = {}
        if target not in self.__targets:
            self.functions[source][target] = partial(func,*args,**kwargs)
            self.__targets[target] = source
        else:
            raise DoubleMappingException
    
    def get_mapping_function(self,source:str,target:str)-> function:
        return self.functions[source][target]
    
    def remove_mapping_function(self,source:str,target:str):
        self.functions[source].pop(target)
        self.__targets.pop(target)
        
    def set_mapping_function(self,source:str, target:str,func:function,*args,**kwargs):
        self.functions[source][target] = partial(func,*args,**kwargs)

                
    def get_sources(self) -> List[str]:
        return self.functions.keys()
    
    def get_targets(self) -> List[str]:
        return self.__targets.keys()
    
class DoubleMappingException(Exception):
    def __init__(self,key):
        super().__init__()
        self.key = key
    
    def __str__(self):
        return "The targed {} can not assigned twice".format(self.key)
    
        