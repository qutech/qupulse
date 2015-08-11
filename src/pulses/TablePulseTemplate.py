"""STANDARD LIBRARY IMPORTS"""
import logging
from typing import Union, Dict, List, Set, Tuple, Optional
import numbers

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from .Parameter import ParameterDeclaration, ImmutableParameterDeclaration, Parameter
from .PulseTemplate import PulseTemplate, MeasurementWindow
from .Sequencer import InstructionBlock, Sequencer
from .Instructions import WaveformTable, Waveform

logger = logging.getLogger(__name__)

class TablePulseTemplate(PulseTemplate):
    """Defines a pulse via linear interpolation of a sequence of (time,voltage)-pairs.
    
    TablePulseTemplate stores a list of (time,voltage)-pairs (the table) which is sorted
    by time and uniquely define a pulse structure via interpolation of voltages of subsequent
    table entries.
    TablePulseTemplate provides methods to declare parameters which may be referred to instead of
    using concrete values for both, time and voltage. If the time of a table entry is a parameter
    reference, it is sorted into the table according to the first value of default, minimum or maximum
    which is defined (not None) in the corresponding ParameterDeclaration. If none of these are defined,
    the entry is placed at the end of the table.
    A TablePulseTemplate may be flagged as representing a measurement pulse, meaning that it defines a
    measurement window.
    """
    
    TableValue = Union[float, ParameterDeclaration]
    TableEntry = Tuple[TableValue, TableValue]
    
    def __init__(self) -> None:
        super().__init__()
        self.__is_sorted = True # type : bool
        self.__entries = [] # type: List[TableEntry]
        self.__time_parameter_declarations = {} # type: Dict[str, ParameterDeclaration]
        self.__voltage_parameter_declarations = {} # type: Dict[str, ParameterDeclaration]
        self.__is_measurement_pulse = False # type: bool
        
    def add_entry(self, time: Union[float, str, ParameterDeclaration], voltage: Union[float, str, ParameterDeclaration]) -> None:
        """Add an entry to this TablePulseTemplate.
        
        The arguments time and voltage may either be real numbers or a string which
        references a parameter declaration by name. If a non-existing parameter declaration
        is referenced, it is created.
        """
        last_entry = (0, 0)
        if self.__entries:
            last_entry = self.__entries[-1]
        
        # construct a ParameterDeclaration if time is a parameter name string
        if isinstance(time, str):
                time = ParameterDeclaration(time)
                
        # if time is (now) a ParameterDeclaration, verify it, insert it and establish references/dependencies to previous entries if necessary
        if isinstance(time, ParameterDeclaration):
            if time.name not in self.__time_parameter_declarations and time.name not in self.__voltage_parameter_declarations:
                if isinstance(time.min_value, ParameterDeclaration):
                    raise ValueError('A ParamterDeclaration for a time parameter may not have a minimum value reference to another ParameterDeclaration object.')
                if isinstance(time.max_value, ParameterDeclaration):
                    raise ValueError('A ParamterDeclaration for a time parameter may not have a maximum value reference to another ParameterDeclaration object.')
                    
                time.min_value = last_entry[0]
                if (isinstance(last_entry[0], ParameterDeclaration) 
                        and (isinstance(last_entry[0].max_value, ParameterDeclaration) or last_entry[0].max_value > time.max_value)): 
                    last_entry[0].max_entry = declaration
                    
                self.__time_parameter_declarations[time.name] = time
            else:
                raise ValueError("A parameter with the name {} already exists.".format(time.name))
        # if time is a real number, ensure that is it not less than the previous entry
        elif isinstance(time, numbers.Real):
            if isinstance(last_entry[0], ParameterDeclaration) and time <= last_entry[0].absolute_max_value or time <= last_entry[0]:
                raise ValueError("Argument time must be greater than previous time value {0}, was: {1}!".format(last_entry[0], time))
                
        # construct a ParameterDeclaration if voltage is a parameter name string
        if isinstance(voltage, str):
            voltage = ParameterDeclaration(voltage)
            
        # if voltage is (now) a ParameterDeclaration, make use of it
        if isinstance(voltage, ParameterDeclaration):
            # check whether a ParameterDeclaration with the same name already exists and, if so, use that instead
            # such that the same object is used consistently for one declaration
            if voltage.name in self.__voltage_parameter_declarations:
                voltage = self.__voltage_parameter_declarations[voltage.name]
            else:
                if voltage.name not in self.__time_parameter_declarations:
                    self.__voltage_parameter_declarations[voltage.name] = voltage
                else:
                    raise ValueError("Argument voltage <{}> must not refer to a time parameter declaration.".format(voltage.name))
            
        # no special action if voltage is a real number
        # finally, add the new entry to the table 
        self.__entries.append((time, voltage))
        
#    def __get_entry_sort_value(self, entry: TableEntry) -> float:
#        """Determine the value of an entry for sorting purposes.
#        
#        If the time value is a constant float, that is returned.
#        If the time value is a parameter reference, the returned value is the first
#         value of default, minimum or maximum which is defined in the parameter declaration.
#         If all these values are None, the result is inf to ensure that the entry will
#         appear at the end of a sorted list.
#         """
#         if isinstance(entry[0]):
#             return entry[0]
#         parameter_declaration = self.__parameter_declarations[entry[0]]
#         if parameter_declaration.default_value is not None:
#             return parameter_declaration.default_value
#         if parameter_declaration.min_value is not None:
#             return parameter_declaration.min_value
#         if parameter_declaration.max_value is not None:
#             return parameter_declaration.max_value
#         return float('inf')
#         
#     def __sort_entries(self) -> None:
#         """Sort this TablePulseTemplate's entries according to their time value.
#         
#         If the time value is a parameter reference it is placed in the sorted list
#         according to the parameter declarations default, minimum or maximum value
#         (with this precedence, i.e., if no default value is given, the minimum value
#         is chosen). If none of these values is set, the entry is appended to the end.
#         """
#         if self.__is_sorted:
#             return
#         self.__entries = sorted(self.__entries, key=self.__get_entry_sort_value)
#         self.__is_sorted = True
        
    @property
    def entries(self) -> List[TableEntry]:
        """Return an immutable copies of this TablePulseTemplate's entries."""
        return self.__entries.copy()
#         entries = []
#         for (time, voltage) in self.__entries:
#             if isinstance(time, ParameterDeclaration):
#                 time = ImmutableParameterDeclaration(time)
#             if isinstance(voltage, ParameterDeclaration):
#                 voltage = ImmutableParameterDeclaration(voltage)
#             entries.append((time, voltage))
#         return entries
        
#    def remove_entry(self, entry: TableEntry) -> None:
#        """Removes an entry from this TablePulseTemplate by its index."""
#        self.__entries.remove(entry)
        
 #   def declare_time_parameter(self, name: str, **kwargs) -> None:
 #       """Declare a new parameter for usage as time value in this TablePulseTemplate.
 #       
 #       If a time parameter declaration for the given name exists, it is overwritten.
 #       
 #       Keyword Arguments:
 #       min -- An optional real number specifying the minimum value allowed for the .
 #       max -- An optional real number specifying the maximum value allowed.
 #       default -- An optional real number specifying a default value for the declared pulse template parameter.
 #       """
 #       self.__time_parameter_declarations[name] = TimeParameterDeclaration(**kwargs)
        
#     def declare_voltage_parameter(self, name:str, **kwargs) -> None:
#         """Declare a new parameter for usage as voltage value in this TablePulseTemplate.
#         
#         If a voltage parameter declaration for the given name exists, it is overwritten.
#         
#         Keyword Arguments:
#         min -- An optional real number specifying the minimum value allowed for the .
#         max -- An optional real number specifying the maximum value allowed.
#         default -- An optional real number specifying a default value for the declared pulse template parameter.
#         """
#         self.__voltage_parameter_declarations[name] = ParameterDeclaration(**kwargs)
    
    def get_parameter_declaration(self, name: str) -> ParameterDeclaration:
        """Return the ParameterDeclaration associated with the given parameter name as an immutable object."""
        if name in self.__time_parameter_declarations:
            return self.__time_parameter_declarations[name]
        elif name in self.__voltage_parameter_declarations:
            return self.__voltage_parameter_declarations[name]
        else:
            raise ParameterNotDeclaredException(name)
        
    def set_is_measurement_pulse(self, is_measurement_pulse: bool) -> None:
        """Set whether or not this TablePulseTemplate represents a measurement pulse."""
        self.__is_measurement_pulse = is_measurement_pulse

    def __str__(self) -> str:
        return __name__

    @property
    def parameter_names(self) -> Set[str]:
        """Return the set of names of declared parameters."""
        return self.__time_parameter_declarations.keys() | self.__voltage_parameter_declarations.keys()

    @property
    def parameter_declarations(self) -> Set[ParameterDeclaration]:
        """Return a set of all parameter declaration objects of this TablePulseTemplate."""
        return self.__time_parameter_declarations.values() | self.__voltage_parameter_declarations.values()
        
    def get_measurement_windows(self, parameters: Optional[Dict[str, Parameter]] = None) -> List[MeasurementWindow]:
        """Return all measurement windows defined in this PulseTemplate.
        
        A TablePulseTemplate specifies either no measurement windows or exactly one that spans its entire duration,
        depending on whether set_is_measurement_pulse(True) was called or not.
        """
        if time_parameters is None:
            raise NotImplementedError()
        
        instantiated_entries = self.__get_entries_instantiated(parameters, None)
        return (0, instantiated_entries[-1][0])

    @property
    def is_interruptable(self) -> bool:
        """Return true, if this PulseTemplate contains points at which it can halt if interrupted."""
        return False
        
    def __get_entries_instantiated(self, parameters: Dict[str, Parameter]) -> List[Tuple[float, float]]:
        """Return a sorted list of all table entries with concrete values provided by the given parameters.
        
        The voltageParameters argument may be None in which case voltage parameter references are not resolved.
        """
        instantiated_entries = [] # type: List[Tuple[float, float]]
        for entry in self.__entries:
            time_value = None # type: float
            voltage_value = None # type: float
            # resolve time parameter references
            if isinstance(entry[0], ParameterDeclaration):
                parameter_declaration = entry[0] # type: ParameterDeclaration
                if not parameter_declaration.check_parameter_set_valid(parameters):
                    raise ParameterValueIllegalException(parameter_declaration, parameter, parameter_declaration)
                
                time_value = parameter_declaration.get_value(parameters)
            else:
                time_value = entry[0]
            # resolve voltage parameter references only if voltageParameters argument is not None, otherwise they are irrelevant
            if isinstance(entry[1], ParameterDeclaration):
                parameter_declaration = entry[1] # type: ParameterDeclaration
                if not parameter_declaration.check_parameter_set_valid(parameters):
                    raise ParameterValueIllegalException(parameter_declaration, parameter, parameter_declaration)
                
                time_value = parameter_declaration.get_value(parameters)
            else:
                voltage_value = entry[1]
            
            instantiated_entries.add((time_value, voltage_value))
            
        return tuple(instantiated_entries)
        
    def build_sequence(self, sequencer: Sequencer, parameters: Dict[str, Parameter], instruction_block: InstructionBlock) -> None:
        waveform = sequencer.register_waveform(self._get_entries_instantiated(parameters))
        instruction_block.add_instruction_exec(waveform)
        
    def requires_stop(self, parameters: Dict[str, Parameter]) -> bool: 
        return any(parameter.requires_stop for parameter in parameters)
    
        
class ParameterDeclarationInUseException(Exception):
    """Indicates that a parameter declaration which should be deleted is in use."""
    
    def __init__(self, declaration_name: str) -> None:
        super.__init__()
        self.declaration_name = declaration_name
        
    def __str__(self) -> str:
        return "The parameter declaration {0} is in use and cannot be deleted.".format(self.declaration_name)
    
    
class ParameterNotDeclaredException(Exception):
    """Indicates that a parameter was not declared."""
    
    def __init__(self, parameter_name: str) -> None:
        super.__init__()
        self.parameter_name = parameter_name
        
    def __str__(self) -> str:
        return "A parameter with the name <{}> was not declared.".format(self.parameter_name)
    
                
class ParameterValueIllegalException(Exception):
    """Indicates that the value provided for a parameter is illegal, i.e., is outside the parameter's bounds or of wrong type."""
    
    def __init__(self, parameter_name: str, parameter: Parameter, parameter_declaration: ParameterDeclaration) -> None:
        super().__init__()
        self.parameter_name = parameter_name
        self.parameter = parameter
        self.parameter_declaration = parameter_declaration
        
    def __str__(self) -> str:
        return "The value {0} provided for parameter {1} is illegal (min = {2}, max = {3})".format(
            self.parameter_name, self.parameter.get_value(), self.parameter_declaration.min_value, self.parameter_declaration.max_value)
