"""STANDARD LIBRARY IMPORTS"""
from logging import getLogger, Logger
from typing import Union, Dict, List, Set, Tuple
import logging

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from pulses.Parameter import ParameterDeclaration, TimeParameterDeclaration, Parameter
from pulses.PulseTemplate import PulseTemplate
from pulses.HardwareUploadInterface import Waveform, PulseHardwareUploadInterface

logger = logging.getLogger(__name__)


TimeTableValue = Union[int, str]
VoltageTableValue = Union[float, str]
TableEntry = Tuple[TimeTableValue, VoltageTableValue]

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
    def __init__(self):
        super().__init__()
        self._isSorted = True # type : bool
        self._entries = [] # type: List[TableEntry]
        self._timeParameterDeclarations = {} # type: Dict[str, TimeParameterDeclaration]
        self._voltageParameterDeclarations = {} # type: Dict[str, ParameterDeclaration]
        self._isMeasurementPulse = False # type: bool
        
    def add_entry(self, time: TimeTableValue, voltage: VoltageTableValue) -> None:
        """!@brief Add an entry to this TablePulseTemplate.
        
        The arguments time and voltage may either be real numbers or a string which
        references a parameter declaration by name. If a non-existing parameter declaration
        is referenced, it is created.
        """
        if isinstance(time, str) and not self._timeParameterDeclarations.has_key(time):
            self.declare_time_parameter(time)
        if isinstance(voltage, str) and not self._voltageParameterDeclarations.has_key(voltage):
            self.declare_voltage_parameter(voltage)
        self._isSorted = False
        self._entries.add((time, voltage))
        
    @staticmethod
    def _get_entry_sort_value(self, entry: TableEntry) -> float:
        """@brief Determine the value of an entry for sorting purposes.
        
        If the time value is a constant float, that is returned.
        If the time value is a parameter reference, the returned value is the first
        value of default, minimum or maximum which is defined in the parameter declaration.
        If all these values are None, the result is inf to ensure that the entry will
        appear at the end of a sorted list.
        """
        if isinstance(entry[0]):
            return entry[0]
        parameterDeclaration = self._parameterDeclarations[entry[0]]
        if parameterDeclaration.defaultValue is not None:
            return parameterDeclaration.defaultValue
        if parameterDeclaration.minValue is not None:
            return parameterDeclaration.minValue
        if parameterDeclaration.maxValue is not None:
            return parameterDeclaration.maxValue
        return float('inf')
        
    def _sort_entries(self) -> None:
        """!@brief Sort this TablePulseTemplate's entries according to their time value.
        
        If the time value is a parameter reference it is placed in the sorted list
        according to the parameter declarations default, minimum or maximum value
        (with this precedence, i.e., if no default value is given, the minimum value
        is chosen). If none of these values is set, the entry is appended to the end.
        """
        if self._isSorted:
            return
        self._entries = sorted(self._entries, key=self._get_entry_sort_value)
        self._isSorted = True
        
    def get_entries(self) -> List[TableEntry]:
        """!@brief Return a sorted copy of this TablePulseTemplate's entries."""
        self._sort_entries()
        return self._entries.copy()
        
    def remove_entry(self, entry: TableEntry) -> None:
        """!@brief Removes an entry from this TablePulseTemplate by its index."""
        self._entries.remove(entry)
        
    def declare_time_parameter(self, name: str, **kwargs) -> None:
        """!@brief Declare a new parameter for usage as time value in this TablePulseTemplate.
        
        If a time parameter declaration for the given name exists, it is overwritten.
        
        Keyword Arguments:
        min -- An optional real number specifying the minimum value allowed for the .
        max -- An optional real number specifying the maximum value allowed.
        default -- An optional real number specifying a default value for the declared pulse template parameter.
        """
        self._timeParameterDeclarations[name] = TimeParameterDeclaration(**kwargs)
        
    def declare_voltage_parameter(self, name:str, **kwargs) -> None:
        """!@brief Declare a new parameter for usage as voltage value in this TablePulseTemplate.
        
        If a voltage parameter declaration for the given name exists, it is overwritten.
        
        Keyword Arguments:
        min -- An optional real number specifying the minimum value allowed for the .
        max -- An optional real number specifying the maximum value allowed.
        default -- An optional real number specifying a default value for the declared pulse template parameter.
        """
        self._voltageParameterDeclarations[name] = ParameterDeclaration(**kwargs)
        
    def get_time_parameter_declaration(self, name: str) -> TimeParameterDeclaration:
        """!@brief Return the TimeParameterDeclaration associated with the given parameter name."""
        return self._timeParameterDeclarations[name]
        
    def get_voltage_parameter_declaration(self, name:str) -> ParameterDeclaration:
        """!@brief Return the voltage ParameterDeclaration associated with the given parameter name."""
        return self._voltageParameterDeclarations[name]
        
    def remove_time_parameter_declaration(self, name: str) -> None:
        """!@brief Remove an existing time parameter declaration from this TablePulseTemplate."""
        for entry in self._entries:
            if (isinstance(entry[0], str)) and (name == entry[0]):
                raise ParameterDeclarationInUseException(name)
        self._timeParameterDeclarations.remove(name)
        
    def remove_voltage_parameter_declaration(self, name: str) -> None:
        """!@brief Remove an existing voltage parameter declaration from this TablePulseTemplate."""
        for entry in self._entries:
            if (isinstance(entry[1], str)) and (name == entry[1]):
                raise ParameterDeclarationInUseException(name)
        self._timeParameterDeclarations.remove(name)
    
    def set_is_measurement_pulse(self, isMeasurementPulse: bool) -> None:
        """!@brief Set whether or not this TablePulseTemplate represents a measurement pulse."""
        self._isMeasurementPulse = isMeasurementPulse
        
    def __len__(self) -> int:
        raise NotImplementedError()
        
    def get_length(self, timeParameters: Dict[str, Parameter]) -> int:
        """!@brief Return the length of the pulse instantiated from this PulseTemplate with the given time parameters."""
        raise NotImplementedError()

    def __str__(self) -> str:
        # TODO: come up with a meaningful description which can be returned here
        return __name__
    
    def get_time_parameter_names(self) -> Set[str]:
        """!@brief Return the set of names of declared time parameters."""
        return self._timeParameterDeclarations.keys()
        
    def get_voltage_parameter_names(self) -> Set[str]:
        """!@brief Return the set of names of declared voltage parameters."""
        return self._voltageParameterDeclarations.keys()
        
    def get_time_parameter_declarations(self) -> Dict[str, TimeParameterDeclaration]:
        """!@brief Return a copy of the dictionary containing the time parameter declarations of this PulseTemplate."""
        return self._timeParameterDeclarations.copy()
        
    def get_voltage_parameter_declarations(self) -> Dict[str, ParameterDeclaration]:
        """!@brief Return a copy of the dictionary containing the voltage parameter declarations of this PulseTemplate."""
        return self._voltageParameterDeclarations.copy()

    def get_measurement_windows(self) -> List[Tuple[float, float]]:
        """!@brief Return all measurement windows defined in this PulseTemplate.
        
        A TablePulseTemplate specifies either no measurement windows or exactly one that spans its entire duration,
        depending on whether set_is_measurement_pulse(True) was called or not.
        """
        if not self._isMeasurementPulse:
            return []
        else:
            return [(0, len(self))] # TODO: will len be defined?

    def is_interruptable(self) -> bool:
        """!@brief Return true, if this PulseTemplate contains points at which it can halt if interrupted."""
        return False

    def upload_waveform(self, uploadInterface: PulseHardwareUploadInterface, parameters: Dict[str, Parameter]) -> Waveform:
        """!@brief Compile a waveform of the pulse represented by this PulseTemplate and the given parameters using the given HardwareUploadInterface object."""
        raise NotImplementedError()
        
    def _get_entries_instantiated(self, timeParameters: Dict[str, Parameter], voltageParameters: Dict[str, Parameter]) -> List[Tuple[int, float]]:
        """!@brief Return a sorted list of all table entries with concrete values provided by the given parameters."""
        instantiatedEntries = [] # type: List[Tuple[int, float]]
        for entry in self._entries:
            timeValue = None # type: int
            voltageValue = None # type: float
            # resolve time parameter references
            if isinstance(entry[0], str):
                parameterDeclaration = self._timeParameterDeclarations[entry[0]] # type: TimeParameterDeclaration
                if timeParameters.has_key(entry[0]):
                    parameter = timeParameters[entry[0]]
                    if not parameterDeclaration.is_parameter_valid(parameter):
                        raise ParameterValueIllegalException(entry[0], parameter, parameterDeclaration)
                    timeValue = parameter.get_value()
                elif parameterDeclaration.defaultValue is not None:
                    timeValue = parameterDeclaration.defaultValue
                else:
                    raise ParameterNotProvidedException(entry[0])
            else:
                timeValue = entry[0]
            # resolve voltage parameter references
            if isinstance(entry[1], str):
                parameterDeclaration = self._voltageParameterDeclarations[entry[1]] # type: ParameterDeclaration
                if voltageParameters.has_key(entry[1]):
                    parameter = voltageParameters[entry[1]]
                    if not parameterDeclaration.is_parameter_valid(parameter):
                        raise ParameterValueIllegalException(entry[1], parameter, parameterDeclaration)
                    voltageValue = parameter.get_value()
                elif parameterDeclaration.defaultValue is not None:
                    voltageValue = parameterDeclaration.defaultValue
                else:
                    raise ParameterNotProvidedException(entry[1])
            else:
                voltageValue = entry[1]
                
            instantiatedEntries.add((timeValue, voltageValue))
            
        # sanity check: no time value must occur more than once
        lastTime = -1 # type: int
        for entry in instantiatedEntries:
            if entry[0] <= lastTime:
                raise DuplicatedTimeEntryException(entry[0])
            lastTime = entry[0]
            
        return sorted(instantiatedEntries)
        
class ParameterDeclarationInUseException(Exception):
    """!@brief Indicated that a parameter declaration which should be deleted is in use."""
    
    def __init__(self, declarationName: str):
        super.__init__()
        self.declarationName = declarationName
        
    def __str__(self):
        return "The parameter declaration {0} is in use and cannot be deleted.".format(self.declarationName)
        
class ParameterNotDeclaredException(Exception):
    """!@brief Indicates that a parameter has not been declared in a TablePulseTemplate."""
    
    def __init__(self, parameterName: str):
        super().__init__()
        self.parameterName = parameterName
        
    def __str__(self):
        return "The parameter {0} has not been declared in the PulseTemplate.".format(self.parameterName)

class ParameterNotProvidedException(Exception):
    """!@brief Indicates that a required parameter value was not provided."""
    
    def __init__(self, parameterName: str):
        super().__init__()
        self.parameterName = parameterName
        
    def __str__(self):
        return "No value was provided for parameter {0} and no default value was specified.".format(self.parameterName)
        
class ParameterValueIllegalException(Exception):
    """!@brief Indicates that the value provided for a parameter is illegal, i.e., is outside the parameter's bounds or of wrong type."""
    
    def __init__(self, parameterName: str, parameter: Parameter, parameterDeclaration: ParameterDeclaration):
        super().__init__()
        self.parameterName = parameterName
        self.parameter = parameter
        self.parameterDeclaration = parameterDeclaration
        
    def __str__(self):
        return "The value {0} provided for parameter {1} is illegal (min = {2}, max = {3})".format(
            self.parameterName, self.parameter.get_value(), self.parameterDeclaration.minValue, self.parameterDeclaration.maxValue)
            
class DuplicatedTimeEntryException(Exception):
    """!@brief Indicates that a time value occurred twice in TablePulseTemplate instantiation."""
    
    def __init__(self, value: int):
        super().__init__()
        self.value = value
        
    def __str__(self):
        return "The time value {0} occurred twice.".format(self.value)