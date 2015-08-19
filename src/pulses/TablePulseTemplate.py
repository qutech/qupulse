"""STANDARD LIBRARY IMPORTS"""
import logging
from typing import Union, Dict, List, Set, Tuple, Optional, Sequence, NamedTuple
import numbers
import numpy as np
from scipy.interpolate import interp1d
import copy
from abc import ABCMeta, abstractmethod

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from .Parameter import ParameterDeclaration, ImmutableParameterDeclaration, Parameter
from .PulseTemplate import PulseTemplate, MeasurementWindow
from .Sequencer import InstructionBlock, Sequencer
from .Instructions import WaveformTable, Waveform

logger = logging.getLogger(__name__)

class InterpolationStrategy(metaclass = ABCMeta):
    
    @abstractmethod
    def __call__(self, start: Tuple[float, float], end: Tuple[float, float], times: np.ndarray) -> np.ndarray:
        """Return a sequence of voltage values for the time slot between the previous and the current point (given as (time, value) pairs)
        according to the interpolation strategy.
        
        The resulting sequence includes the sample for the time of the current point and start at the sample just after the previous point, i.e., 
        is of the form [f(sample(previous_point_time)+1), f(sample(previous_point_time)+2), ... f(sample(current_point_time))].
        """
        pass
    
class LinearInterpolationStrategy(InterpolationStrategy):
    """Interpolates linearly."""
    
    def __call__(self, start: Tuple[float, float], end: Tuple[float, float], times: np.ndarray) -> np.ndarray:
        xs = [start[0], end[0]]
        ys = [start[1], end[1]]

        interpolator = interp1d(xs, ys, kind='linear', copy=False) # No extra error checking needed, interp1d throws errors for times out of bounds
        return interpolator(times)

    def __repr__(self):
        return "<Linear Interpolation>"

    
class HoldInterpolationStrategy(InterpolationStrategy):
    """Holds previous value and jumps to the current value at the last sample."""

    def __call__(self, start: Tuple[float, float], end: Tuple[float, float], times: np.ndarray) -> np.ndarray:
        xs = [start[0], end[0]]
        ys = [start[1], end[1]]

        if np.any(times < start[0]) or np.any(times > end[0]):
            raise ValueError("Time Value for interpolation out of bounds. Must be between {0} and {1}.".format(start[0], end[0]))

        voltages = np.ones_like(times) * start[1]
        return voltages

    def __repr__(self):
        return "<Hold Interpolation>"

class JumpInterpolationStrategy(InterpolationStrategy):
    """Jumps to the current value at the first sample and holds."""
    # TODO: better name than jump

    def __call__(self, start: Tuple[float, float], end: Tuple[float, float], times: np.ndarray) -> np.ndarray:
        xs = [start[0], end[0]]
        ys = [start[1], end[1]]

        if np.any(times < start[0]) or np.any(times > end[0]):
           raise ValueError("Time Value for interpolation out of bounds. Must be between {0} and {1}.".format(start[0], end[0]))

        voltages = np.ones_like(times) * end[1]
        return voltages

    def __repr__(self):
        return "<Jump Interpolation>"


TableValue = Union[float, ParameterDeclaration]
TableEntry = NamedTuple("TableEntry", [('t', TableValue), ('v', TableValue), ('interp', InterpolationStrategy)])

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

    def __init__(self) -> None:
        super().__init__()
        self.__entries = [] # type: List[TableEntry]
        self.__time_parameter_declarations = {} # type: Dict[str, ParameterDeclaration]
        self.__voltage_parameter_declarations = {} # type: Dict[str, ParameterDeclaration]
        self.__is_measurement_pulse = False # type: bool
        self.__interpolation_strategies = {
                                           'linear': LinearInterpolationStrategy(), 
                                           'hold': HoldInterpolationStrategy(), 
                                           'jump': JumpInterpolationStrategy()
                                          }
        
    def add_entry(self, 
                  time: Union[float, str, ParameterDeclaration], 
                  voltage: Union[float, str, ParameterDeclaration], 
                  interpolation: str = 'hold') -> None:
        """Add an entry to this TablePulseTemplate.
        
        The arguments time and voltage may either be real numbers or a string which
        references a parameter declaration by name or a ParameterDeclaration object.
        The following constraints hold:
        - If a non-existing parameter declaration is referenced (via string), it is created without min, max and default values.
        - Parameter declarations for the time domain may not be used multiple times. Else a ValueError is thrown.
        - ParameterDeclaration objects for the time domain may not refer to other ParameterDeclaration objects as min or max values. Else a ValueError is thrown.
        - If a ParameterDeclaration is provided, its min and max values are set to its neighboring values if they were not set previously.
        - Each entries time value must be greater than its predecessor's, i.e.,
            - if the time value is a float and the previous time value is a float, the new value must be greater
            - if the time value is a float and the previous time value is a parameter declaration, the new value must not be smaller than the maximum of the parameter declaration
            - if the time value is a parameter declaration and the previous time value is a float, the new values minimum must be no smaller
            - if the time value is a parameter declaration and the previous time value is a parameter declaration, the new minimum must not be smaller than the previous minimum
              and the previous maximum must not be greater than the new maximum
        """
        if not self.__entries:
            # if the first entry has a time that is either > 0 or a parameter declaration, insert a start point (0, 0)
            if not isinstance(time, numbers.Real) or time > 0:
                self.__entries.append(TableEntry(0, 0, self.__interpolation_strategies['linear'])) # interpolation strategy for first entry is disregarded, could be anything
                last_entry = self.__entries[-1]
            # ensure that the first entry is not negative
            elif isinstance(time, numbers.Real) and time < 0:
                raise ValueError("Time value must not be negative, was {}".format(time))
            elif time == 0:
                last_entry = (-1,0)
        else:
            last_entry = self.__entries[-1]
        
        # construct a ParameterDeclaration if time is a parameter name string
        if isinstance(time, str):
                time = ParameterDeclaration(time)
                
        # if time is (now) a ParameterDeclaration, verify it, insert it and establish references/dependencies to previous entries if necessary
        if isinstance(time, ParameterDeclaration):
            if time.name in self.__voltage_parameter_declarations:
                raise ValueError("Cannot use already declared voltage parameter '{}' in time domain.".format(time.name))
            if time.name not in self.__time_parameter_declarations:
                if isinstance(time.min_value, ParameterDeclaration):
                    raise ValueError("A ParameterDeclaration for a time parameter may not have a minimum value reference" \
                                     " to another ParameterDeclaration object.")
                if isinstance(time.max_value, ParameterDeclaration):
                    raise ValueError("A ParameterDeclaration for a time parameter may not have a maximum value reference" \
                                     " to another ParameterDeclaration object.")
                    
                # set minimum value if not previously set
                if time.min_value == float('-inf'):
                    time.min_value = last_entry[0]

                # TODO: check these cases. There are a lot of bugs here, like 'declaration' not being defined and comparing
                # parameters to integers is also a problem. We need unit tests
                #if isinstance(last_entry[0], ParameterDeclaration):
                #    # set maximum value of previous entry if not already set
                #    #if last_entry[0].max_value == float('+inf'):
                #    #    last_entry[0].max_entry = declaration

                #    if time.min_value < last_entry[0].absolute_min_value:
                #        raise ValueError("Argument time's minimum value must be no smaller than the previous time" \
                #                         "parameter declaration's minimum value. Parameter '{0}', Minimum {1}, Provided {2}."
                #                         .format(last_entry[0].name, last_entry[0].absolute_min_value, time.min_value))
                #    if time.max_value < last_entry[1].absolute_max_value:
                #        raise ValueError("Argument time's maximum value must be no smaller than the previous time" \
                #                         " parameter declaration's maximum value. Parameter '{0}', Maximum {1}, Provided {2}."
                #                         .format(last_entry[0].name, last_entry[0].absolute_max_value, time.max_value))
                    
                self.__time_parameter_declarations[time.name] = time
            else:
                raise ValueError("A time parameter with the name {} already exists.".format(time.name))
        # if time is a real number, ensure that is it not less than the previous entry
        elif isinstance(time, numbers.Real):
            if isinstance(last_entry[0], ParameterDeclaration):
                # set maximum value of previous entry if not already set
                if last_entry[0].max_value == float('+inf'):
                    last_entry[0].max_entry = time
                    
                if time < last_entry[0].absolute_max_value:
                    raise ValueError("Argument time must be no smaller than previous time parameter declaration's" \
                                     " maximum value. Parameter '{0}', Maximum {1}, Provided: {2}."
                                     .format(last_entry[0].name, last_entry[0].absolute_max_value, time))
                
            elif time <= last_entry[0]:
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
        try:
            interpolation_strategy = self.__interpolation_strategies[interpolation]
        except:
            raise ValueError("The interpolation strategy '{}' is unknown.".format(interpolation))
        self.__entries.append(TableEntry(time, voltage, interpolation_strategy))
        
    @property
    def entries(self) -> List[TableEntry]:
        """Return an immutable copies of this TablePulseTemplate's entries."""
        return copy.deepcopy(self.__entries)

    @property
    def parameter_names(self) -> Set[str]:
        """Return the set of names of declared parameters."""
        return set(self.__time_parameter_declarations.keys()) | set(self.__voltage_parameter_declarations.keys())

    @property
    def parameter_declarations(self) -> Set[ParameterDeclaration]:
        """Return a set of all parameter declaration objects of this TablePulseTemplate."""
        return set(self.__time_parameter_declarations.values()) | set(self.__voltage_parameter_declarations.values())

    def get_measurement_windows(self, parameters: Optional[Dict[str, Parameter]] = None) -> List[MeasurementWindow]: # TODO: not very robust
        """Return all measurement windows defined in this PulseTemplate.
        
        A TablePulseTemplate specifies either no measurement windows or exactly one that spans its entire duration,
        depending on whether set_is_measurement_pulse(True) was called or not.
        """

        if not self.__is_measurement_pulse:
            return []
        else:
            if parameters is None:
                raise NotImplementedError()

            instantiated_entries = self.get_entries_instantiated(parameters)
            return [(0, instantiated_entries[-1][0])]
    
    @property
    def is_interruptable(self) -> bool:
        """Return true, if this PulseTemplate contains points at which it can halt if interrupted."""
        return False
        
    def get_entries_instantiated(self, parameters: Dict[str, Parameter]) -> List[Tuple[float, float, InterpolationStrategy]]:
        """Return a list of all table entries with concrete values provided by the given parameters.
        """
        instantiated_entries = [] # type: List[Tuple[float, float]]
        for entry in self.__entries:
            time_value = None # type: float
            voltage_value = None # type: float
            # resolve time parameter references
            if isinstance(entry.t, ParameterDeclaration):
                parameter_declaration = entry.t # type: ParameterDeclaration
                if not parameter_declaration.check_parameter_set_valid(parameters):
                    raise ParameterValueIllegalException(parameter_declaration, parameter, parameter_declaration)
                
                time_value = parameter_declaration.get_value(parameters)
            else:
                time_value = entry.t
            # resolve voltage parameter references only if voltageParameters argument is not None, otherwise they are irrelevant
            if isinstance(entry.v, ParameterDeclaration):
                parameter_declaration = entry.v # type: ParameterDeclaration
                if not parameter_declaration.check_parameter_set_valid(parameters):
                    raise ParameterValueIllegalException(parameter_declaration, parameter, parameter_declaration)
                
                voltage_value= parameter_declaration.get_value(parameters)
            else:
                voltage_value = entry.v
            
            instantiated_entries.append(TableEntry(time_value, voltage_value, entry.interp))
            
        # ensure that no time value occurs twice
        previous_time = -1
        for (time, _, _) in instantiated_entries:
            if time <= previous_time:
                raise Exception("Time value {0} is smaller than the previous value {1}.".format(time, previous_time))
            previous_time = time
            
        return instantiated_entries

    def render(self, parameters: Dict[str, Parameter], ts: np.ndarray):
        """Instantiates using parameters and evaluates the voltage curve at times ts."""
        instantiated = self.get_entries_instantiated(parameters)
        voltages = np.empty_like(ts) # prepare voltage vector
        for entry1, entry2 in zip(instantiated[:-1], instantiated[1:]): # iterate over interpolated areas
            indices = np.logical_and(ts >= entry1.t, ts <= entry2.t)
            voltages[indices] = entry2.interp(entry1, entry2, ts[indices]) # evaluate interpolation at each time
        return voltages

    def build_sequence(self, sequencer: Sequencer, parameters: Dict[str, Parameter], instruction_block: InstructionBlock) -> None:
        waveform = sequencer.register_waveform(self.get_entries_instantiated(parameters))
        instruction_block.add_instruction_exec(waveform)
        
    def requires_stop(self, parameters: Dict[str, Parameter]) -> bool: 
        return any(parameter.requires_stop for parameter in parameters.values())
    
        
class ParameterDeclarationInUseException(Exception):
    """Indicates that a parameter declaration which should be deleted is in use."""
    
    def __init__(self, declaration_name: str) -> None:
        super().__init__()
        self.declaration_name = declaration_name
        
    def __str__(self) -> str:
        return "The parameter declaration {0} is in use and cannot be deleted.".format(self.declaration_name)
    
    
class ParameterNotDeclaredException(Exception):
    """Indicates that a parameter was not declared."""
    
    def __init__(self, parameter_name: str) -> None:
        super().__init__()
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
            self.parameter_name, self.parameter.get_value(), self.parameter_declaration.min_value,
            self.parameter_declaration.max_value)
