"""STANDARD LIBRARY IMPORTS"""
from logging import getLogger, Logger
from typing import Union, Dict, List, Set, Tuple
import logging

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from Parameter import ParameterDeclaration, Parameter
from PulseTemplate import PulseTemplate

logger = logging.getLogger(__name__)


TableValue = Union[float, str]
TableEntry = Tuple[Union[float, str], Union[float, str]]

class Table(PulseTemplate):
    """docstring for Table"""
    def __init__(self):
        super().__init__()
        self._isSorted = True
        self._entries = [] # type: List[TableEntry]
        self._parameterDeclarations = {} # type: Dict[str, ParameterDeclaration]
        self._isMeasurementPulse = False
        
    def add_entry(self, time: TableValue, voltage: TableValue) -> None:
        """Add an entry to this Table pulse template.
        
        The arguments time and voltage may either be real numbers or a string which
        references a parameter declaration by name. If a non-existing parameter declaration
        is referenced, this method raises a ParameterNotDeclaredException.
        """
        if isinstance(time, str) and not self._parameterDeclarations.has_key(time):
            raise ParameterNotDeclaredException(time)
        if isinstance(voltage, str) and not self._parameterDeclarations.has_key(voltage):
            raise ParameterNotDeclaredException(voltage)
        self._isSorted = False
        self._entries.add((time, voltage))
        
    @staticmethod
    def _get_entry_sort_value(self, entry: TableEntry) -> float:
        """Determine the value of an entry for sorting purposes.
        
        If the time value is a constant float, that is returned.
        If the time value is a parameter reference the returned value is the first
        value of default, minimum or maximum which is defined in the parameter declaration.
        If all these values are None, the result is len(self)+1 to ensure that the entry will
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
        return len(self)+1
        
    def _sort_entries(self) -> None:
        """Sort this Table's entries according to their time value.
        
        If the time value is a parameter reference it is placed in the sorted list
        according to the parameter declarations default, minimum or maximum value
        (with this precedence, i.e., if no default value is given, the minimum value
        is chosen). If none of these values is set, the entry is appended to the end.
        """
        if self._isSorted:
            return
        self._entries = sorted(self._entries, key=Table._get_entry_sort_value)
        self._isSorted = True
        
    def get_entries(self) -> List[TableEntry]:
        """Return a sorted copy of this Table's entries."""
        self._sort_entries()
        return self._entries.copy()
        
    def remove_entry(self, entry: TableEntry) -> None:
        """Removes an entry from this Table pulse template by its index."""
        self._entries.remove(entry)
        
    def declare_parameter(self, name: str, **kwargs) -> None:
        """Declare a new parameter for use in this Table.
        
        If a parameter declaration for the given name exists, it is overwritten.
        
        Keyword Arguments:
        min or minValue -- An optional real number specifying the minimum value allowed for the .
        max or maxValue -- An optional real number specifying the maximum value allowed.
        default or defaultValue -- An optional real number specifying a default value for the declared pulse template parameter.
        """
        self._parameterDeclarations[name] = ParameterDeclaration(**kwargs)
        
    def remove_parameter_declaration(self, name: str) -> None:
        """Remove an existing parameter declaration from this Table."""
        # TODO: check whether the parameter declaration is referenced from entries and delete if not
        raise NotImplementedError()
    
    def set_is_measurement_pulse(self, isMeasurementPulse: bool) -> None:
        """Set whether or not this Table represents a measurement pulse."""
        self._isMeasurementPulse = isMeasurementPulse
        
    def __len__(self) -> int:
        """Defines the behaviour of len(PulseTemplate), which is the sum of all subpulses. 
        __len__ already provides a type check to assure that only numerical values are returned
        """
        raise NotImplementedError()

    def __str__(self) -> str:
        # TODO: come up with a meaningful description which can be returned here
        raise NotImplementedError()
    
    def get_parameter_names(self) -> Set[str]:
        """Return the set of names of declared parameters."""
        return self._parameterDeclarations.keys()
        
    def get_parameter_declarations(self) -> Dict[str, ParameterDeclaration]:
        """Return a copy of the dictionary containing the parameter declarations of this PulseTemplate."""
        return self._parameterDeclarations.copy()

    def get_measurement_windows(self) -> List[Tuple[float, float]]:
        """Return all measurment windows defined in this PulseTemplate.
        
        A Table specifies either no measurement windows or exactly one that spans its entire duration,
        depending on whether set_is_measurement_pulse(True) was called or not.
        """
        if not self._isMeasurementPulse:
            return []
        else:
            return [(0, len(self))]

    def is_interruptable(self) -> bool:
        """Return true, if this PulseTemplate contains points at which it can halt if interrupted."""
        return False

    def generate_waveforms(self, sequencer: "Sequencer", parameters: Dict[str, Parameter]) -> None:
        """Compile a waveform of the pulse represented by this PulseTemplate and the given parameters using the hardware-specific Sequencer object."""
        # TODO: pending detailed definition of Sequencer class
        raise NotImplementedError()
        
class ParameterNotDeclaredException(Exception):
    """Indicates that a parameter has not been declared in a Table pulse template."""
    
    def __init__(self, parameterName: str):
        super().__init__()
        self._parameterName = parameterName
        
    def __str__(self):
        return "The parameter {0} has not been declared in the PulseTemplate.".format(self._parameterName)
