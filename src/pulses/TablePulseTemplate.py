"""STANDARD LIBRARY IMPORTS"""
from logging import getLogger, Logger
from typing import Union, Dict, List, Set, Tuple
import logging

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from pulses.Parameter import ParameterDeclaration, TimeParameterDeclaration, Parameter
from pulses.PulseTemplate import PulseTemplate, MeasurementWindow
from pulses.Sequencer import InstructionBlock, Sequencer
from pulses.Instructions import WaveformTable, Waveform

logger = logging.getLogger(__name__)

class TablePulseTemplate(PulseTemplate):
    """!@brief Defines a pulse via linear interpolation of a sequence of (time,voltage)-pairs.
    
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
    
    TimeValue = Union[int, str]
    VoltageValue = Union[float, str]
    TableEntry = Tuple[TimeValue, VoltageValue]
    
    def __init__(self):
        super().__init__()
        self.__is_sorted = True # type : bool
        self.__entries = [] # type: List[TableEntry]
        self.__time_parameter_declarations = {} # type: Dict[str, TimeParameterDeclaration]
        self.__voltage_parameter_declarations = {} # type: Dict[str, ParameterDeclaration]
        self.__is_measurement_pulse = False # type: bool
        
    def add_entry(self, time: TimeValue, voltage: VoltageValue) -> None:
        """!@brief Add an entry to this TablePulseTemplate.
        
        The arguments time and voltage may either be real numbers or a string which
        references a parameter declaration by name. If a non-existing parameter declaration
        is referenced, it is created.
        """
        if isinstance(time, str):
            if time not in self.__time_parameter_declarations:
                self.declare_time_parameter(time)
        else:
            if time < 0:
                raise ValueError("Argument time must be positive, was: {}!".format(time))
                
        if isinstance(voltage, str) and voltage not in self.__voltage_parameter_declarations:
            self.declare_voltage_parameter(voltage)
        self.__is_sorted = False
        self.__entries.add((time, voltage))
        
    def __get_entry_sort_value(self, entry: TableEntry) -> float:
        """@brief Determine the value of an entry for sorting purposes.
        
        If the time value is a constant float, that is returned.
        If the time value is a parameter reference, the returned value is the first
        value of default, minimum or maximum which is defined in the parameter declaration.
        If all these values are None, the result is inf to ensure that the entry will
        appear at the end of a sorted list.
        """
        if isinstance(entry[0]):
            return entry[0]
        parameter_declaration = self.__parameter_declarations[entry[0]]
        if parameter_declaration.default_value is not None:
            return parameter_declaration.default_value
        if parameter_declaration.min_value is not None:
            return parameter_declaration.min_value
        if parameter_declaration.max_value is not None:
            return parameter_declaration.max_value
        return float('inf')
        
    def __sort_entries(self) -> None:
        """!@brief Sort this TablePulseTemplate's entries according to their time value.
        
        If the time value is a parameter reference it is placed in the sorted list
        according to the parameter declarations default, minimum or maximum value
        (with this precedence, i.e., if no default value is given, the minimum value
        is chosen). If none of these values is set, the entry is appended to the end.
        """
        if self.__isSorted:
            return
        self.__entries = sorted(self.__entries, key=self.__get_entry_sort_value)
        self.__isSorted = True
        
    def get_entries(self) -> List[TableEntry]:
        """!@brief Return a sorted copy of this TablePulseTemplate's entries."""
        self.__sort_entries()
        return self.__entries.copy()
        
    def remove_entry(self, entry: TableEntry) -> None:
        """!@brief Removes an entry from this TablePulseTemplate by its index."""
        self.__entries.remove(entry)
        
    def declare_time_parameter(self, name: str, **kwargs) -> None:
        """!@brief Declare a new parameter for usage as time value in this TablePulseTemplate.
        
        If a time parameter declaration for the given name exists, it is overwritten.
        
        Keyword Arguments:
        min -- An optional real number specifying the minimum value allowed for the .
        max -- An optional real number specifying the maximum value allowed.
        default -- An optional real number specifying a default value for the declared pulse template parameter.
        """
        self.__time_parameter_declarations[name] = TimeParameterDeclaration(**kwargs)
        
    def declare_voltage_parameter(self, name:str, **kwargs) -> None:
        """!@brief Declare a new parameter for usage as voltage value in this TablePulseTemplate.
        
        If a voltage parameter declaration for the given name exists, it is overwritten.
        
        Keyword Arguments:
        min -- An optional real number specifying the minimum value allowed for the .
        max -- An optional real number specifying the maximum value allowed.
        default -- An optional real number specifying a default value for the declared pulse template parameter.
        """
        self.__voltage_parameter_declarations[name] = ParameterDeclaration(**kwargs)
        
    def get_time_parameter_declaration(self, name: str) -> TimeParameterDeclaration:
        """!@brief Return the TimeParameterDeclaration associated with the given parameter name."""
        return self.__time_parameter_declarations[name]
        
    def get_voltage_parameter_declaration(self, name:str) -> ParameterDeclaration:
        """!@brief Return the voltage ParameterDeclaration associated with the given parameter name."""
        return self.__voltage_parameter_declarations[name]
        
    def remove_time_parameter_declaration(self, name: str) -> None:
        """!@brief Remove an existing time parameter declaration from this TablePulseTemplate."""
        for entry in self.__entries:
            if (isinstance(entry[0], str)) and (name == entry[0]):
                raise ParameterDeclarationInUseException(name)
        self.__time_parameter_declarations.remove(name)
        
    def remove_voltage_parameter_declaration(self, name: str) -> None:
        """!@brief Remove an existing voltage parameter declaration from this TablePulseTemplate."""
        for entry in self.__entries:
            if (isinstance(entry[1], str)) and (name == entry[1]):
                raise ParameterDeclarationInUseException(name)
        self.__voltage_parameter_declarations.remove(name)
    
    def set_is_measurement_pulse(self, is_measurement_pulse: bool) -> None:
        """!@brief Set whether or not this TablePulseTemplate represents a measurement pulse."""
        self.__is_measurement_pulse = is_measurement_pulse

    def __str__(self) -> str:
        return __name__
    
    def get_time_parameter_names(self) -> Set[str]:
        """!@brief Return the set of names of declared time parameters."""
        return self.__time_parameter_declarations.keys()
        
    def get_voltage_parameter_names(self) -> Set[str]:
        """!@brief Return the set of names of declared voltage parameters."""
        return self.__voltage_parameter_declarations.keys()
        
    def get_time_parameter_declarations(self) -> Dict[str, TimeParameterDeclaration]:
        """!@brief Return a copy of the dictionary containing the time parameter declarations of this PulseTemplate."""
        return self.__time_parameter_declarations.copy()
        
    def get_voltage_parameter_declarations(self) -> Dict[str, ParameterDeclaration]:
        """!@brief Return a copy of the dictionary containing the voltage parameter declarations of this PulseTemplate."""
        return self.__voltage_parameter_declarations.copy()
        
    def get_measurement_windows(self, time_parameters: Dict[str, Parameter] = None) -> List[MeasurementWindow]:
        """!@brief Return all measurement windows defined in this PulseTemplate.
        
        A TablePulseTemplate specifies either no measurement windows or exactly one that spans its entire duration,
        depending on whether set_is_measurement_pulse(True) was called or not.
        """
        if time_parameters is None:
            raise NotImplementedError()
        
        instantiated_entries = self.__get_entries_instantiated(time_parameters, None)
        return (0, instantiated_entries[-1][0])

    def is_interruptable(self) -> bool:
        """!@brief Return true, if this PulseTemplate contains points at which it can halt if interrupted."""
        return False
        
    def __get_entries_instantiated(self, time_parameters: Dict[str, Parameter], voltage_parameters: Dict[str, Parameter] = None) -> List[Tuple[int, VoltageValue]]:
        """!@brief Return a sorted list of all table entries with concrete values provided by the given parameters.
        
        The voltageParameters argument may be None in which case voltage parameter references are not resolved.
        """
        instantiated_entries = [] # type: List[Tuple[int, VoltageValue]]
        for entry in self.__entries:
            time_value = None # type: int
            voltage_value = None # type: VoltageValue
            # resolve time parameter references
            if isinstance(entry[0], str):
                parameter_declaration = self.__time_parameter_declarations[entry[0]] # type: TimeParameterDeclaration
                if entry[0] in time_parameters:
                    parameter = time_parameters[entry[0]]
                    if not parameter_declaration.is_parameter_valid(parameter):
                        raise ParameterValueIllegalException(entry[0], parameter, parameter_declaration)
                    time_value = parameter.get_value()
                elif parameter_declaration.default_value is not None:
                    time_value = parameter_declaration.defaultValue
                else:
                    raise ParameterNotProvidedException(entry[0])
            else:
                time_value = entry[0]
            # resolve voltage parameter references only if voltageParameters argument is not None, otherwise they are irrelevant
            if isinstance(entry[1], str) and voltage_parameters is not None:
                parameter_declaration = self.__voltage_parameter_declarations[entry[1]] # type: ParameterDeclaration
                if entry[1] in voltage_parameters:
                    parameter = voltage_parameters[entry[1]]
                    if not parameter_declaration.is_parameter_valid(parameter):
                        raise ParameterValueIllegalException(entry[1], parameter, parameter_declaration)
                    voltage_value = parameter.get_value()
                elif parameter_declaration.default_value is not None:
                    voltage_value = parameter_declaration.default_value
                else:
                    raise ParameterNotProvidedException(entry[1])
            else:
                voltage_value = entry[1]
            
            instantiated_entries.add((time_value, voltage_value))
            
        # sanity check: no time value must occur more than once
        last_time = -1 # type: int
        for entry in instantiated_entries:
            if entry[0] <= last_time:
                raise DuplicatedTimeEntryException(entry[0])
            last_time = entry[0]
            
        return tuple(sorted(instantiated_entries))
        
    def build_sequence(self, sequencer: Sequencer, time_parameters: Dict[str, Parameter], voltage_parameters: Dict[str, Parameter], instruction_block: InstructionBlock) -> None:
        waveform = sequencer.register_waveform(self._get_entries_instantiated(time_parameters, voltage_parameters))
        instruction_block.add_instruction_exec(waveform)
        
class ParameterDeclarationInUseException(Exception):
    """!@brief Indicates that a parameter declaration which should be deleted is in use."""
    
    def __init__(self, declaration_name: str):
        super.__init__()
        self.declaration_name = declaration_name
        
    def __str__(self):
        return "The parameter declaration {0} is in use and cannot be deleted.".format(self.declaration_name)
        
class ParameterNotDeclaredException(Exception):
    """!@brief Indicates that a parameter has not been declared in a TablePulseTemplate."""
    
    def __init__(self, parameter_name: str):
        super().__init__()
        self.parameter_name = parameter_name
        
    def __str__(self):
        return "The parameter {0} has not been declared in the PulseTemplate.".format(self.parameter_name)

class ParameterNotProvidedException(Exception):
    """!@brief Indicates that a required parameter value was not provided."""
    
    def __init__(self, parameter_name: str):
        super().__init__()
        self.parameter_name = parameter_name
        
    def __str__(self):
        return "No value was provided for parameter {0} and no default value was specified.".format(self.parameter_name)
        
class ParameterValueIllegalException(Exception):
    """!@brief Indicates that the value provided for a parameter is illegal, i.e., is outside the parameter's bounds or of wrong type."""
    
    def __init__(self, parameter_name: str, parameter: Parameter, parameter_declaration: ParameterDeclaration):
        super().__init__()
        self.parameter_name = parameter_name
        self.parameter = parameter
        self.parameter_declaration = parameter_declaration
        
    def __str__(self):
        return "The value {0} provided for parameter {1} is illegal (min = {2}, max = {3})".format(
            self.parameter_name, self.parameter.get_value(), self.parameter_declaration.min_value, self.parameter_declaration.max_value)
            
class DuplicatedTimeEntryException(Exception):
    """!@brief Indicates that a time value occurred twice in TablePulseTemplate instantiation."""
    
    def __init__(self, value: int):
        super().__init__()
        self.value = value
        
    def __str__(self):
        return "The time value {0} occurred twice.".format(self.value)