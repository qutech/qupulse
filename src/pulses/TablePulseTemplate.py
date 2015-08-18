"""STANDARD LIBRARY IMPORTS"""
import logging
from typing import Union, Dict, List, Set, Tuple, Optional, Sequence
import numbers
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
    def interpolate(self, previous_point: Tuple[float, float], current_point: Tuple[float, float], samples_per_microsecond: float) -> Sequence[float]:
        """Return a sequence of voltage values for the time slot between the previous and the current point (given as (time, value) pairs)
        according to the interpolation strategy.
        
        The resulting sequence includes the sample for the time of the current point and start at the sample just after the previous point, i.e., 
        is of the form [f(sample(previous_point_time)+1), f(sample(previous_point_time)+2), ... f(sample(current_point_time))].
        """
        pass
    
class LinearInterpolationStrategy(InterpolationStrategy):
    """Interpolates linearly."""
    
    def interpolate(self, previous_point: Tuple[float, float], current_point: Tuple[float, float], samples_per_microsecond: float) -> Sequence[float]:
        (p_time, p_value) = previous_point
        (c_time, c_value) = current_point
        
        duration = c_time - p_time
        if duration <= 0:
            raise ValueError("Duration between points must be > 0, was {}".format(duration))
        
        samples = []
        time_per_sample = 1.0 / samples_per_microsecond
        for time in range(0, duration, time_per_sample):
            samples.append( (duration-time) * p_value + time * c_value )
        
        return samples
    
class HoldInterpolationStrategy(InterpolationStrategy):
    """Holds previous value and jumps to the current value at the last sample."""
    
    def interpolate(self, previous_point: Tuple[float, float], current_point: Tuple[float, float], samples_per_microsecond: float) -> Sequence[float]:
        (p_time, p_value) = previous_point
        (c_time, c_value) = current_point
        
        duration = c_time - p_time
        if duration <= 0:
            raise ValueError("Duration between points must be > 0, was {}".format(duration))
        
        sample_count = samples_per_microsecond * duration
        samples = [p_values] * sample_count
        samples[-1] = c_value
        
        return samples
    
class JumpInterpolationStrategy(InterpolationStrategy):
    """Jumps to the current value at the first sample and holds."""
    # TODO: better name than jump
    
    def interpolate(self, previous_point: Tuple[float, float], current_point: Tuple[float, float], samples_per_microsecond: float) -> Sequence[float]:
        (p_time, p_value) = previous_point
        (c_time, c_value) = current_point
        
        duration = c_time - p_time
        if duration <= 0:
            raise ValueError("Duration between points must be > 0, was {}".format(duration))
        
        sample_count = samples_per_microsecond * duration
        samples = [c_value] * sample_count
        
        return samples
    

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
    TableEntry = Tuple[TableValue, TableValue, InterpolationStrategy]
    
    def __init__(self) -> None:
        super().__init__()
        self.__entries = [] # type: List[TableEntry]
        self.__time_parameter_declarations = {} # type: Dict[str, ParameterDeclaration]
        self.__voltage_parameter_declarations = {} # type: Dict[str, ParameterDeclaration]
        self.__is_measurement_pulse = False # type: bool
        self.__interpolation_strategies = {'linear': LinearInterpolationStrategy(), 'hold': HoldInterpolationStrategy(), 'jump': JumpInterpolationStrategy()}
        
    def add_entry(self, time: Union[float, str, ParameterDeclaration], voltage: Union[float, str, ParameterDeclaration], interpolation: str = 'hold') -> None:
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
                self.__entries.append((0, 0, self.__interpolation_strategies['linear'])) # interpolation strategy for first entry is disregarded, could be anything
            # ensure that the first entry is not negative
            elif isinstance(time, numbers.Real) and time < 0:
                raise ValueError("Time value must not be negative, was {}".format(time))
        
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
                    raise ValueError('A ParamterDeclaration for a time parameter may not have a minimum value reference to another ParameterDeclaration object.')
                if isinstance(time.max_value, ParameterDeclaration):
                    raise ValueError('A ParamterDeclaration for a time parameter may not have a maximum value reference to another ParameterDeclaration object.')
                    
                # set minimum value if not previously set
                if time.min_value == float('-inf'):
                    time.min_value = last_entry[0]
                    
                if isinstance(last_entry[0], ParameterDeclaration):
                    # set maximum value of previous entry if not already set
                    if last_entry[0].max_value == float('+inf'): 
                        last_entry[0].max_entry = declaration
                        
                    if time.min_value < last_entry[0].absolute_min_value:
                        raise ValueError("Argument time's minimum value must be no smaller than the previous time parameter declaration's minimum value. Parameter '{0}', Minimum {1}, Provided {2}."
                                         .format(last_entry[0].name, last_entry[0].absolute_min_value, time.min_value))
                    if time.max_value < last_entry[1].absolute_max_value:
                        raise ValueError("Argument time's maximum value must be no smaller than the previous time parameter declaration's maximum value. Parameter '{0}', Maximum {1}, Provided {2}."
                                         .format(last_entry[0].name, last_entry[0].absolute_max_value, time.max_value))
                    
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
                    raise ValueError("Argument time must be no smaller than previous time parameter declaration's maximum value. Parameter '{0}', Maximum {1}, Provided: {2}."
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
        self.__entries.append((time, voltage, interpolation_strategy))
        
    @property
    def entries(self) -> List[TableEntry]:
        """Return an immutable copies of this TablePulseTemplate's entries."""
        entries = []
        for (time, value, _) in self.__entries:
            entries.append((time, value))
        return entries
            
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
        if parameters is None:
            raise NotImplementedError()
        
        instantiated_entries = self.get_entries_instantiated(parameters, None)
        return [(0, instantiated_entries[-1][0])]    
    
    @property
    def is_interruptable(self) -> bool:
        """Return true, if this PulseTemplate contains points at which it can halt if interrupted."""
        return False
        
    def get_entries_instantiated(self, parameters: Dict[str, Parameter]) -> List[Tuple[float, float]]:
        """Return a list of all table entries with concrete values provided by the given parameters.
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
            
            instantiated_entries.append((time_value, voltage_value))
            
        # ensure that no time value occurs twice
        previous_time = -1
        for (time, _) in instantiated_entries:
            if time <= previous_time:
                raise Exception("Time value {0} is smaller than the previous value {1}.".format(time, previous_time))
            previous_time = time
            
        return tuple(instantiated_entries)
    
    def get_interpolated_voltage_values(self, parameters: Dict[str, Parameter], samples_per_microsecond: float) -> Sequence[float]:
        instantiated_entries = self.get_entries_instantiated(parameters)
        if not instantiated_entries: 
            return []
        
        voltage_values = []
        voltage_values[0] = instantiated_entries[0][1]
        for i in range(1, len(instantiated_entries)):
            previous_point = instantiated_entries[i - 1]
            current_point = instantiated_entries[i]
            (_, _, interpolation_strategy) = self.__entries[i]
            voltage_values.extend(interpolation_strategy.interpolate(prevoius_point, current_point, samples_per_microsecond))
            
        
    def build_sequence(self, sequencer: Sequencer, parameters: Dict[str, Parameter], instruction_block: InstructionBlock) -> None:
        waveform = sequencer.register_waveform(self.get_entries_instantiated(parameters))
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
