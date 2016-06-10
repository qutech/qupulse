"""This module defines the TablePulseTemplate, one of the elementary pulse templates and its
waveform representation.

Classes:
    - TablePulseTemplate: Defines a pulse via interpolation of a sequence of (time,voltage)-pairs.
    - TableWaveform: A waveform instantiated from a TablePulseTemplate by providing values for its
        declared parameters.
"""

from typing import Union, Dict, List, Set, Optional, NamedTuple, Any, Iterable, Tuple
import numbers
import copy

import numpy as np

from qctoolkit.serialization import Serializer
from qctoolkit.pulses.parameters import ParameterDeclaration, Parameter, \
    ParameterNotProvidedException
from qctoolkit.pulses.pulse_template import PulseTemplate, MeasurementWindow
from qctoolkit.pulses.sequencing import InstructionBlock, Sequencer
from qctoolkit.pulses.interpolation import InterpolationStrategy, LinearInterpolationStrategy, \
    HoldInterpolationStrategy, JumpInterpolationStrategy
from qctoolkit.pulses.instructions import Waveform
from qctoolkit.pulses.conditions import Condition

__all__ = ["TablePulseTemplate", "TableWaveform", "WaveformTableEntry"]

WaveformTableEntry = NamedTuple( # pylint: disable=invalid-name
    "WaveformTableEntry",
    [('t', float), ('v', float), ('interp', InterpolationStrategy)]
)


class TableWaveform(Waveform):
    """Waveform obtained from instantiating a TablePulseTemplate."""

    def __init__(self, waveform_tables: List[Tuple[WaveformTableEntry]]) -> None:
        """Create a new TableWaveform instance.

        Args:
            waveform_table (ImmutableList(WaveformTableEntry)): A list of instantiated table
                entries of the form (time as float, voltage as float, interpolation strategy).
        """
        for table in waveform_tables:
            if len(table) < 2:
                raise ValueError("The given WaveformTable has less than two entries.")
        super().__init__()
        self.__tables = waveform_tables

    @property
    def compare_key(self) -> Any:
        return tuple(map(tuple, self.__tables))

    @property
    def channels(self) -> int:
        return len(self.__tables)

    @property
    def duration(self) -> float:
        if len(self.__tables):
            return max([table[-1].t for table in self.__tables])
        else:
            return 0.0

    def sample(self, sample_times: np.ndarray, first_offset: float=0) -> np.ndarray:
        channels = len(self.__tables)
        sample_times = sample_times.copy()
        sample_times -= (sample_times[0] - first_offset)
        voltages = np.empty((len(sample_times), channels))
        for channel, table in enumerate(self.__tables):
            for entry1, entry2 in zip(table[:-1], table[1:]):
                indices = np.logical_and(sample_times >= entry1.t, sample_times <= entry2.t)
                voltages[indices, channel] = \
                    entry2.interp((entry1.t, entry1.v), (entry2.t, entry2.v), sample_times[indices])

        return voltages


TableValue = Union[float, ParameterDeclaration] # pylint: disable=invalid-name
TableEntry = NamedTuple( # pylint: disable=invalid-name
    "TableEntry",
    [('t', TableValue), ('v', TableValue), ('interp', InterpolationStrategy)]
)


class TablePulseTemplate(PulseTemplate):
    """Defines a pulse via interpolation of a sequence of (time,voltage)-pairs.

    TablePulseTemplate stores a list of (time,voltage)-pairs (the table) which is sorted
    by time and uniquely define a pulse structure via interpolation of voltages of subsequent
    table entries.
    TablePulseTemplate provides methods to declare parameters which may be referred to instead of
    using concrete values for both, time and voltage.
    A TablePulseTemplate may be flagged as representing a measurement pulse, meaning that it defines
    a measurement window.
    """

    def __init__(self, channels=1, measurement=False, identifier: Optional[str]=None) -> None:
        """Create a new TablePulseTemplate.

        Args:
            measurement (bool): True, if this TablePulseTemplate shall define a measurement window.
                (optional, default = False).
            identifier (str): A unique identifier for use in serialization. (optional)
        """
        super().__init__(identifier)
        self.__identifier = identifier
        self.__entries = []
        for _ in range(channels):
            self.__entries.append([]) # type: List[List[TableEntry]]
        self.__time_parameter_declarations = {} # type: Dict[str, ParameterDeclaration]
        self.__voltage_parameter_declarations = {} # type: Dict[str, ParameterDeclaration]
        self.__is_measurement_pulse = measurement# type: bool
        self.__interpolation_strategies = {'linear': LinearInterpolationStrategy(),
                                           'hold': HoldInterpolationStrategy(), 
                                           'jump': JumpInterpolationStrategy()
                                          }
        self.__channels = channels # type: int

    @staticmethod
    def from_array(times: np.ndarray, voltages: np.ndarray, measurement=False) \
            -> 'TablePulseTemplate':
        """Static constructor to build a TablePulse from numpy arrays.

        Args:
            times: 1D numpy array with time values
            voltages: 1D or 2D numpy array with voltage values

        Returns:
            TablePulseTemplate with the given values, hold interpolation everywhere and no free
            parameters.
        """
        if voltages.ndim == 1:
            res = TablePulseTemplate(channels=1, measurement=measurement)
            for time, voltage in zip(times, voltages):
                res.add_entry(time, voltage, interpolation='hold')
        elif voltages.ndim == 2:
            channels = voltages.shape[1]
            res = TablePulseTemplate(channels=channels, measurement=measurement)
            for channel in range(channels):
                for time, voltage in zip(times, voltages[:, channel]):
                    res.add_entry(time, voltage, interpolation='hold', channel=channel)
        return res

    def add_entry(self,
                  time: Union[float, str, ParameterDeclaration], 
                  voltage: Union[float, str, ParameterDeclaration], 
                  interpolation: str='hold',
                  channel: int=0) -> None:
        """Add an entry to the end of this TablePulseTemplate.

        The arguments time and voltage may either be real numbers or a string which
        references a parameter declaration by name or a ParameterDeclaration object.

        If the first entry provided to the table has a time > 0, a (0,0) entry is automatically
        inserted in front.

        The following constraints hold:
        - If a non-existing parameter declaration is referenced (via string), it is created without
            min, max and default values.
        - Parameter declarations for the time domain may not be used multiple times.
        - ParameterDeclaration objects for the time domain may not refer to other
            ParameterDeclaration objects as min or max values.
        - If a ParameterDeclaration is provided, its min and max values will be set to its
            neighboring values if they were not set previously or would exceed neighboring bounds.
        - Each entries time value must be greater than its predecessor's, i.e.,
            - if the time value is a float and the previous time value is a float, the new value
                must be greater
            - if the time value is a float and the previous time value is a parameter declaration,
                the new value must not be smaller than the maximum of the parameter declaration
            - if the time value is a parameter declaration and the previous time value is a float,
                the new values minimum must be no smaller
            - if the time value is a parameter declaration and the previous time value is a
                parameter declaration, the new minimum must not be smaller than the previous minimum
                and the previous maximum must not be greater than the new maximum

        Args:
            time (float or str or ParameterDeclaration): The time value of the new entry. Either a
                constant real number or some parameter name/declaration.
            voltage (float or str or ParameterDeclaration): The voltage value of the new entry.
                Either a constant real number or some parameter name/declaration.
            interpolation (str): The interpolation strategy between the previously last and the new
                entry. One of 'linear', 'hold' (hold previous value) or 'jump' (jump immediately
                to new value).
            channel (int): The channel in which the voltage value will be set. (default = 0)
        Raises:
            ValueError if the constraints listed above are violated.
        """
        # Check if channel is valid
        if channel < 0 or channel > self.__channels - 1:
            raise ValueError("Channel number out of bounds. Allowed values: {}".format(
                ', '.join(map(str, range(self.__channels))))
            )

        # Check if interpolation value is valid
        if interpolation not in self.__interpolation_strategies.keys():
            raise ValueError("Interpolation strategy not implemented. Allowed values: {}."
                             .format(', '.join(self.__interpolation_strategies.keys())))
        else:
            interpolation = self.__interpolation_strategies[interpolation]

        entries = self.__entries[channel]
        # If this is the first entry, there are a number of cases we have to check
        if not entries:
            # if the first entry has a time that is either > 0 or a parameter declaration,
            # insert a start point (0, 0)
            if not isinstance(time, numbers.Real) or time > 0:
                last_entry = TableEntry(0, 0, self.__interpolation_strategies['hold'])
            # ensure that the first entry is not negative
            elif isinstance(time, numbers.Real) and time < 0:
                raise ValueError("Time value must not be negative, was {}.".format(time))
            elif time == 0.0:
                last_entry = TableEntry(-1, 0, self.__interpolation_strategies['hold'])
        else:
            last_entry = entries[-1]

        # Handle time parameter/value
        time = self.__add_entry_check_and_modify_time(time, last_entry)

        # Handle voltage parameter/value
        # construct a ParameterDeclaration if voltage is a parameter name string
        if isinstance(voltage, str):
            voltage = ParameterDeclaration(voltage)

        # if voltage is (now) a ParameterDeclaration, make use of it
        if isinstance(voltage, ParameterDeclaration):
            # check whether a ParameterDeclaration with the same name already exists and, if so,
            # use that instead such that the same object is used consistently for one declaration
            if voltage.name in self.__voltage_parameter_declarations:
                voltage = self.__voltage_parameter_declarations[voltage.name]
            elif (voltage.name in self.__time_parameter_declarations or
                  (isinstance(time, ParameterDeclaration) and voltage.name == time.name)):
                raise ValueError(
                    "Argument voltage <{}> must not refer to a time parameter declaration."
                        .format(voltage.name)
                )

        # no special action if voltage is a real number

        # add declarations to declaration sets if necessary
        if isinstance(time, ParameterDeclaration):
            self.__time_parameter_declarations[time.name] = time
        if isinstance(voltage, ParameterDeclaration):
            self.__voltage_parameter_declarations[voltage.name] = voltage
        # in case we need a time 0 entry previous to the new entry
        if not entries and (not isinstance(time, numbers.Real) or time > 0):
            entries.append(last_entry)
        # finally, add the new entry to the table 
        entries.append(TableEntry(time, voltage, interpolation))
        self.__entries[channel] = entries

    def __add_entry_check_and_modify_time(self,
                                          time: Union[float, str, Parameter],
                                          last_entry: TableValue) -> TableValue:
        # Handle time parameter/value

        # first case: time is a real number
        if isinstance(time, numbers.Real):
            if isinstance(last_entry.t, ParameterDeclaration):
                # set maximum value of previous entry if not already set
                if last_entry.t.max_value == float('+inf'):
                    last_entry.t.max_value = time

                if time < last_entry.t.absolute_max_value:
                    raise ValueError(
                        "Argument time must be no smaller than previous time parameter "
                        " declaration's maximum value. Parameter '{0}', Maximum {1}, Provided: {2}."
                        .format(
                            last_entry.t.name, last_entry.t.absolute_max_value, time
                        )
                    )

            # if time is a real number, ensure that is it not less than the previous entry
            elif time <= last_entry.t:
                raise ValueError(
                    "Argument time must be greater than previous time value {0}, was: {1}!".format(
                        last_entry.t, time
                    )
                )

        # second case: time is a string -> Create a new ParameterDeclaration and continue third case
        elif isinstance(time, str):
            time = ParameterDeclaration(time)

        # third case: time is a ParameterDeclaration
        # if time is (now) a ParameterDeclaration, verify it, insert it and establish
        # references/dependencies to previous entries if necessary
        if isinstance(time, ParameterDeclaration):
            if time.name in self.__voltage_parameter_declarations:
                raise ValueError(
                    "Cannot use already declared voltage parameter '{}' in time domain.".format(
                        time.name
                    )
                )
            if time.name not in self.__time_parameter_declarations:
                if isinstance(time.min_value, ParameterDeclaration):
                    raise ValueError(
                        "A ParameterDeclaration for a time parameter may not have a minimum value "
                        "reference to another ParameterDeclaration object."
                    )
                if isinstance(time.max_value, ParameterDeclaration):
                    raise ValueError(
                        "A ParameterDeclaration for a time parameter may not have a maximum value "
                        "reference to another ParameterDeclaration object."
                    )

                # make a (shallow) copy of the ParameterDeclaration to ensure that it can't be
                # changed from outside the table
                time = ParameterDeclaration(time.name, min=time.min_value, max=time.max_value,
                                            default=time.default_value)
                # set minimum value if not previously set
                # if last_entry.t is a ParameterDeclaration, its max_value field will be set
                # accordingly by the min_value setter of the new entry, ensuring a correct boundary
                # relationship between both declarations
                if time.min_value == float('-inf'):
                    time.min_value = last_entry.t

                # Check dependencies between successive time parameters
                if isinstance(last_entry.t, ParameterDeclaration):

                    if last_entry.t.max_value == float('inf'):
                        last_entry.t.max_value = time

                    if time.absolute_min_value < last_entry.t.absolute_min_value:
                        raise ValueError(
                            "Argument time's minimum value must be no smaller than the previous "
                            "time parameter declaration's minimum value. Parameter '{0}', Minimum "
                            "{1}, Provided {2}.".format(
                                last_entry.t.name, last_entry.t.absolute_min_value, time.min_value
                            )
                        )
                    if time.absolute_max_value < last_entry.t.absolute_max_value:
                        raise ValueError(
                            "Argument time's maximum value must be no smaller than the previous "
                            "time parameter declaration's maximum value. Parameter '{0}', Maximum "
                            "{1}, Provided {2}.".format(
                                last_entry.t.name, last_entry.t.absolute_max_value, time.max_value
                            )
                        )
                else:
                    if time.min_value < last_entry.t:
                        raise ValueError(
                            "Argument time's minimum value {0} must be no smaller than the previous"
                            " time value {1}.".format(time.min_value, last_entry.t)
                        )
            else:
                raise ValueError(
                    "A time parameter with the name {} already exists.".format(time.name)
                )
        return time
        
    @property
    def entries(self) -> List[List[TableEntry]]:
        """Immutable copies of this TablePulseTemplate's entries."""
        return copy.deepcopy(self.__entries)

    @property
    def parameter_names(self) -> Set[str]:
        """The set of names of declared parameters."""
        return set(self.__time_parameter_declarations.keys()) \
               | set(self.__voltage_parameter_declarations.keys())

    @property
    def parameter_declarations(self) -> Set[ParameterDeclaration]:
        """A set of all parameter declaration objects of this TablePulseTemplate."""
        return set(self.__time_parameter_declarations.values()) | \
               set(self.__voltage_parameter_declarations.values())

    def get_measurement_windows(self,
                                parameters: Optional[Dict[str, Parameter]]=None) \
            -> List[MeasurementWindow]: # TODO: remove
        if not self.__is_measurement_pulse:
            return []
        else:
            # instantiated_entries = self.get_entries_instantiated(parameters)
            last_entries = [channel[-1].t for channel in self.get_entries_instantiated(parameters)]
            maxtime = max(last_entries)
            return [(0, maxtime)]

    @property
    def is_interruptable(self) -> bool:
        return False
        
    def get_entries_instantiated(self, parameters: Dict[str, Parameter]) \
            -> List[List[Tuple[float, float]]]:
        """Compute an instantiated list of the table's entries.

        Args:
            parameters (Dict(str -> Parameter)): A mapping of parameter names to Parameter objects.
        Returns:
             (float, float)-list of all table entries with concrete values provided by the given
                parameters.
        """
        instantiated_entries = [] # type: List[List[TableEntry]]
        for channel in range(self.__channels):
            entries = self.__entries[channel]
            instantiated = []
            for entry in entries:
                # resolve time parameter references
                if isinstance(entry.t, ParameterDeclaration):
                    time_value = entry.t.get_value(parameters)
                else:
                    time_value = entry.t
                if isinstance(entry.v, ParameterDeclaration):
                    voltage_value = entry.v.get_value(parameters)
                else:
                    voltage_value = entry.v

                instantiated.append(TableEntry(time_value, voltage_value, entry.interp))

            # ensure that no time value occurs twice
            previous_time = -1
            for (time, _, _) in instantiated:
                if time <= previous_time:
                    raise Exception("Time value {0} is smaller than the previous value {1}."
                                    .format(time, previous_time))
                previous_time = time

            instantiated_entries.append(instantiated)
        return list(map(TablePulseTemplate.__clean_entries, instantiated_entries))

    @staticmethod
    def __clean_entries(entries: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """ Checks if three subsequent values in a list of table entries have the same value.
        If so, the intermediate is redundant and removed in-place.

        Args:
            entries (List(TableEntry)): List of table entries to clean. Will be modified in-place.
        Returns:
            a reference to entries
        """
        length = len(entries)
        if not entries or length < 3:
            return entries

        for index in range(length-2, 0, -1):
            previous_step = entries[index - 1]
            step = entries[index]
            next_step = entries[index + 1]
            if step.v == previous_step.v and step.v == next_step.v:
                entries.pop(index)
        return entries

    def build_sequence(self,
                       sequencer: Sequencer,
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, Condition],
                       instruction_block: InstructionBlock) -> None:
        instantiated = self.get_entries_instantiated(parameters)
        if instantiated[0]:
            waveform = TableWaveform(instantiated)
            instruction_block.add_instruction_exec(waveform)

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, 'Condition']) -> bool:
        try:
            return any(
                parameters[name].requires_stop
                for name in self.parameter_names
                if not isinstance(parameters[name], numbers.Number)
            )
        except KeyError as key_error:
            raise ParameterNotProvidedException(str(key_error)) from key_error

    def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
        data = dict()
        data['is_measurement_pulse'] = self.__is_measurement_pulse
        data['time_parameter_declarations'] = \
            [serializer.dictify(self.__time_parameter_declarations[key])
             for key in sorted(self.__time_parameter_declarations.keys())]
        data['voltage_parameter_declarations'] = \
            [serializer.dictify(self.__voltage_parameter_declarations[key])
             for key in sorted(self.__voltage_parameter_declarations.keys())]
        serialized_entries = []
        for channel in self.__entries:
            entries = []
            for (time, voltage, interpolation) in channel:
                if isinstance(time, ParameterDeclaration):
                    time = time.name
                if isinstance(voltage, ParameterDeclaration):
                    voltage = voltage.name
                entries.append((time, voltage, str(interpolation)))
            serialized_entries.append(entries)
        data['entries'] = serialized_entries
        data['type'] = serializer.get_type_identifier(self)
        return data

    @staticmethod
    def deserialize(serializer: Serializer,
                    time_parameter_declarations: Iterable[Any],
                    voltage_parameter_declarations: Iterable[Any],
                    entries: Iterable[Any],
                    is_measurement_pulse: bool,
                    identifier: Optional[str]=None) -> 'TablePulseTemplate':
        time_parameter_declarations = \
            {declaration['name']: serializer.deserialize(declaration)
             for declaration in time_parameter_declarations}
        voltage_parameter_declarations = \
            {declaration['name']: serializer.deserialize(declaration)
             for declaration in voltage_parameter_declarations}

        template = TablePulseTemplate(is_measurement_pulse, identifier=identifier)

        for channel, channelentries in enumerate(entries):
            for (time, voltage, interpolation) in channelentries:
                if isinstance(time, str):
                    time = time_parameter_declarations[time]
                if isinstance(voltage, str):
                    voltage = voltage_parameter_declarations[voltage]
                template.add_entry(time, voltage, interpolation=interpolation, channel=channel)

        return template
