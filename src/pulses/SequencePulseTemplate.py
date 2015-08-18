"""STANDARD LIBRARY IMPORTS"""
import logging
from typing import Dict, List, Tuple, Set, Callable
from functools import partial
"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""

from .PulseTemplate import PulseTemplate, MeasurementWindow
from .Parameter import ParameterDeclaration, Parameter


logger = logging.getLogger(__name__)

class SequencePulseTemplate(PulseTemplate):
    """A sequence of different PulseTemplates.
    
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
    
    def __init__(self) -> None:
        super().__init__()
        self.subtemplates = []  # type: List[PulseTemplate]
        self.mapping = Mapping()  # type: Mapping
        self.__is_interruptable = None
    
    @property
    def parameter_names(self) -> Set[str]:
        parameter_names = set()
        for subtemplate in self.subtemplates:
            for parameter_name in subtemplate.parameter_names:
                if parameter_name in self.mapping.get_targets():
                    parameter_names.add(self.mapping.get_targets()[parameter_name])
                else:
                    parameter_names.add(parameter_name)
        return parameter_names
                    
    @property
    def parameter_declarations(self) -> Set[ParameterDeclaration]:
        parameter_declarations = set()
        for subtemplate in self.subtemplates:
            for declaration in subtemplate.parameter_declarations:
                if declaration.name in self.mapping.get_targets():
                    #Rename the parameter according to the mapping. The target name becomes the source name.
                    declaration = ParameterDeclaration(
                                                       self.mapping.get_targets()[declaration.name], 
                                                       min=declaration.min_value, 
                                                       max=declaration.max_value, 
                                                       default=declaration.default_value
                                                       ) 
                    # TODO: min, max and default values might have to be mapped to? especially in the case that min, max are ParameterDeclarations as well
                parameter_declarations.add(declaration)
        return parameter_declarations
          
    def get_measurement_windows(self, parameters: Dict[str, Parameter] = None) -> List[MeasurementWindow]:
        measurement_windows = []
        for subtemplate in self.subtemplates:
            # TODO: parameters might have to be mapped
            measurement_windows = measurement_windows + subtemplate.get_measurement_windows(parameters)
        return measurement_windows
    
    @property
    def is_interruptable(self) -> bool:
        for subtemplate in self.subtemplates:
            if not subtemplate.is_interruptable():
                return False
        return True

    def get_mapping(self) -> Mapping:
        return self.mapping
    
class Mapping(object):
    def __init__(self) -> None:
        super().__init__()
        self.functions = {}  # type: Dict[str, Dict[str, Callable]]
        self.__targets = {}  # type: Dict[str, str]
    
    def add_mapping_function(self,source:str,target:str,func:Callable,*args,**kwargs) -> None:
        if source not in self.functions:
            self.functions[source] = {}
        if target not in self.__targets:
            self.functions[source][target] = partial(func,*args,**kwargs)
            self.__targets[target] = source
        else:
            raise DoubleMappingException(target)
    
    def get_mapping_function(self,source:str,target:str)-> Callable:
        return self.functions[source][target]
    
    def remove_mapping_function(self,source:str,target:str) -> None:
        self.functions[source].pop(target)
        self.__targets.pop(target)
        
    def set_mapping_function(self,source:str, target:str,func:Callable,*args,**kwargs) -> None:
        self.functions[source][target] = partial(func,*args,**kwargs)

                
    def get_sources(self) -> List[str]:
        return self.functions.keys()
    
    def get_targets(self) -> List[str]:
        return self.__targets.keys()
    
class DoubleMappingException(Exception):
    def __init__(self,key) -> None:
        super().__init__()
        self.key = key
    
    def __str__(self) -> str:
        return "The targed {} can not assigned twice".format(self.key)
    
        