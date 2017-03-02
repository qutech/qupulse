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

from qctoolkit import MeasurementWindow, ChannelID
from qctoolkit.serialization import Serializer
from qctoolkit.pulses.parameters import ParameterDeclaration, Parameter, \
    ParameterNotProvidedException
from qctoolkit.pulses.pulse_template import AtomicPulseTemplate
from qctoolkit.pulses.interpolation import InterpolationStrategy, LinearInterpolationStrategy, \
    HoldInterpolationStrategy, JumpInterpolationStrategy
from qctoolkit.pulses.instructions import Waveform
from qctoolkit.pulses.conditions import Condition
from qctoolkit.expressions import Expression
from qctoolkit.pulses.multi_channel_pulse_template import MultiChannelWaveform

__all__ = ["TablePulseTemplate", "TableWaveform", "WaveformTableEntry"]

WaveformTableEntry = NamedTuple( # pylint: disable=invalid-name
    "WaveformTableEntry",
    [('t', float), ('v', float), ('interp', InterpolationStrategy)]
)


class TableWaveform(Waveform):
    """Waveform obtained from instantiating a TablePulseTemplate."""

    def __init__(self,
                 channel: ChannelID,
                 waveform_table: Iterable[WaveformTableEntry],
                 measurement_windows: Iterable[MeasurementWindow]) -> None:
        """Create a new TableWaveform instance.

        Args:
            waveform_table (ImmutableList(WaveformTableEntry)): A list of instantiated table
                entries of the form (time as float, voltage as float, interpolation strategy).
        """
        if len(waveform_table) < 2:
            raise ValueError("A given waveform table has less than two entries.")
        super().__init__()
        self.__table = tuple(waveform_table)
        self.__channel_id = channel
        self.__measurement_windows = tuple(measurement_windows)

    @property
    def compare_key(self) -> Any:
        return self.__channel_id, self.__table, self.__measurement_windows

    @property
    def duration(self) -> float:
        return self.__table[-1].t

    def unsafe_sample(self,
                      channel: ChannelID,
                      sample_times: np.ndarray,
                      output_array: Union[np.ndarray, None]=None) -> np.ndarray:
        if output_array is None:
            output_array = np.empty(len(sample_times))

        for entry1, entry2 in zip(self.__table[:-1], self.__table[1:]):
            indices = slice(np.searchsorted(sample_times, entry1.t, 'left'),
                            np.searchsorted(sample_times, entry2.t, 'right'))
            output_array[indices] = \
                entry2.interp((entry1.t, entry1.v), (entry2.t, entry2.v), sample_times[indices])
        return output_array

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return {self.__channel_id}

    def get_measurement_windows(self) -> Iterable[MeasurementWindow]:
        return self.__measurement_windows

    def unsafe_get_subset_for_channels(self, channels: Set[ChannelID]) -> 'Waveform':
        return self


TableValue = Union[float, ParameterDeclaration] # pylint: disable=invalid-name
TableEntry = NamedTuple( # pylint: disable=invalid-name
    "TableEntry",
    [('t', TableValue), ('v', TableValue), ('interp', InterpolationStrategy)]
)
MeasurementDeclaration = Tuple[Union[float,Expression], Union[float,Expression]]


class TablePulseTemplate(AtomicPulseTemplate):
    """Defines a pulse via interpolation of a sequence of (time,voltage)-pairs.

    TablePulseTemplate stores a list of (time,voltage)-pairs (the table) which is sorted
    by time and uniquely define a pulse structure via interpolation of voltages of subsequent
    table entries.
    TablePulseTemplate provides methods to declare parameters which may be referred to instead of
    using concrete values for both, time and voltage.
    A TablePulseTemplate may be flagged as representing a measurement pulse, meaning that it defines
    a measurement window.

    Each TablePulseTemplate contains at least an entry at time 0.
    """

    def __init__(self, channels: List[ChannelID] = ['default'], identifier: Optional[str]=None) -> None:
        """Create a new TablePulseTemplate.

        Args:
            channels (int): The list of channel identifiers defined in this TablePulseTemplate (default = 1).
            measurement (bool): True, if this TablePulseTemplate shall define a measurement window.
                (optional, default = False).
            identifier (str): A unique identifier for use in serialization. (optional)
        """
        if len(set(channels)) != len(channels):
            raise ValueError('ChannelIDs must be unique')

        super().__init__(identifier)
        self.__identifier = identifier
        self.__interpolation_strategies = {'linear': LinearInterpolationStrategy(),
                                           'hold': HoldInterpolationStrategy(),
                                           'jump': JumpInterpolationStrategy()
                                           }
        self.__entries = dict((channel, [TableEntry(0, 0, self.__interpolation_strategies['hold'])])
                              for channel in channels)
        self.__time_parameter_declarations = {} # type: Dict[str, ParameterDeclaration]
        self.__voltage_parameter_declarations = {} # type: Dict[str, ParameterDeclaration]
        self.__measurement_windows = {} # type: Dict[str,List[MeasurementDeclaration]]

    @staticmethod
    def from_array(times: np.ndarray, voltages: np.ndarray, channels: Optional[List[ChannelID]] = None) \
            -> 'TablePulseTemplate':
        """Static constructor to build a TablePulse from numpy arrays.

        Args:
            times: 1D numpy array with time values
            voltages: 1D or 2D numpy array with voltage values
            channels: list of channel IDs. Mandatory if voltages is 2D

        Returns:
            TablePulseTemplate with the given values, hold interpolation everywhere and no free
            parameters.
        """
        res = TablePulseTemplate(channels=channels) if channels else TablePulseTemplate()
        if voltages.ndim == 1:
            for time, voltage in zip(times, voltages):
                res.add_entry(time, voltage, interpolation='hold')
        elif voltages.ndim == 2:
            if not channels:
                raise ValueError('For a multi channel table pulse template a list of channel IDs mut be provided.')
            if len(channels) != voltages.shape[1]:
                raise ValueError('There has to be exactly one channel ID for each channel.')
            for channel_index in range(len(channels)):
                for time, voltage in zip(times, voltages[:, channel_index]):
                    res.add_entry(time, voltage, interpolation='hold', channel=channels[channel_index])
        return res

    def add_entry(self,
                  time: Union[float, str, ParameterDeclaration], 
                  voltage: Union[float, str, ParameterDeclaration], 
                  interpolation: str='hold',
                  channel: Optional[ChannelID]=None) -> None:
        """Add an entry to the end of this TablePulseTemplate.

        The arguments time and voltage may either be real numbers or a string which
        references a parameter declaration by name or a ParameterDeclaration object.

        If the first entry provided to the table has a time > 0, a (0,0) entry is automatically
        inserted in front.

        The following constraints hold:
        - If a non-existing parameter declaration is referenced (via string), it is created without
            min, max and default values.
        - Parameter declarations for the time domain may not be used multiple times in the same
            channel.
        - If a ParameterDeclaration is provided, its min and max values will be set to its
            neighboring values if they were not set previously or would exceed neighboring bounds.
        - Parameter declarations for the time domain used in different channels will have their
            bounds set such that range conforms with any channel automatically.
        - ParameterDeclaration objects for the time domain may not refer to other
            ParameterDeclaration objects as min or max values.
        - Each entries time value must be greater than its predecessor's, i.e.,
            - if the time value is a float and the previous time value is a float, the new value
                must be greater or equal
            - if the time value is a float and the previous time value is a parameter declaration
                and the new value is smaller then the maximum of the parameter declaration, the
                maximum is adjusted to the new value
            - if the time value is a float and the previous time value is a parameter declaration,
                the new value must not be smaller than the minimum of the parameter declaration
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
        if channel is None:
            if len(self.__entries) == 1:
                channel = next(iter(self.__entries.keys()))
            else:
                raise ValueError('Channel ID has to be specified if more than one channel is present')
        elif channel not in self.__entries:
            raise ValueError("Channel ID not known. Allowed values: {}".format(
                ', '.join(self.__entries.keys()))
            )

        # Check if interpolation value is valid
        if interpolation not in self.__interpolation_strategies.keys():
            raise ValueError("Interpolation strategy not implemented. Allowed values: {}."
                             .format(', '.join(self.__interpolation_strategies.keys())))
        else:
            interpolation = self.__interpolation_strategies[interpolation]

        entries = self.__entries[channel]

        last_entry = entries[-1]
        # Handle time parameter/value
        time = self.__add_entry_check_and_modify_time(time, entries)

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
        new_entry = TableEntry(time, voltage, interpolation)
        if last_entry.t == time:
            entries[-1] = new_entry
        else:
            entries.append(new_entry)
        self.__entries[channel] = entries

    def __add_entry_check_and_modify_time(self,
                                          time: Union[float, str, Parameter],
                                          entries: List[TableEntry]) -> TableValue:
        last_entry = entries[-1]
        # Handle time parameter/value

        # first case: time is a real number
        if isinstance(time, numbers.Real):
            if isinstance(last_entry.t, ParameterDeclaration):
                # set maximum value of previous entry if not already set
                if last_entry.t.max_value == float('+inf'):
                    last_entry.t.max_value = time

                if time < last_entry.t.absolute_max_value:
                    try:
                        last_entry.t.max_value = time
                    except ValueError:
                        raise ValueError(
                            "Argument time must not be smaller than the minimum of the previous"
                            "parameter declaration ({}, was {}).".format(
                                last_entry.t.absolute_min_value,
                                time
                            )
                        )

            # if time is a real number, ensure that is it not less than the previous entry
            elif time < last_entry.t:
                raise ValueError(
                    "Argument time must not be less than previous time value {0}, was: {1}!".format(
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
            if time.name in [e.t.name for e in entries if isinstance(e.t, ParameterDeclaration)]:
                raise ValueError(
                    "A time parameter with the name {} already exists.".format(time.name)
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
            else:
                time = self.__time_parameter_declarations[time.name]

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
        return time
        
    @property
    def entries(self) -> Union[List[TableEntry],Dict[ChannelID,List[TableEntry]]]:
        """Immutable copies of this TablePulseTemplate's entries."""
        if len(self.__entries) == 1:
            return copy.deepcopy(next(iter(self.__entries.values())))
        else:
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
                                parameters: Dict[str, Parameter],
                                measurement_mapping: Dict[str, str]) -> List[MeasurementWindow]:
        def get_val(v):
            return v if not isinstance(v, Expression) else v.evaluate_numeric(
              **{name_: parameters[name_].get_value() if isinstance(parameters[name_], Parameter) else parameters[name_]
              for name_ in v.variables()})

        t_max = [entry[-1][0] for entry in self.__entries.values()]
        t_max = max([t if isinstance(t,numbers.Number) else t.get_value(parameters) for t in t_max])

        resulting_windows = []
        for name, windows in self.__measurement_windows.items():
            for begin, end in windows:
                resulting_windows.append((measurement_mapping[name], get_val(begin), get_val(end)))
                if resulting_windows[-1][2] > t_max:
                    raise ValueError('Measurement window out of pulse')
        return resulting_windows

    @property
    def measurement_declarations(self):
        """
        :return: Measurement declarations as added by the add_measurement_declaration method
        """
        as_builtin = lambda x: str(x) if isinstance(x, Expression) else x
        return {name: [(as_builtin(begin), as_builtin(end))
                       for begin, end in windows]
                for name, windows in self.__measurement_windows.items() }

    @property
    def measurement_names(self) -> Set[str]:
        """
        :return:
        """
        return set(self.__measurement_windows.keys())

    def add_measurement_declaration(self, name: str, begin: Union[float,str], end: Union[float,str]) -> None:
        if isinstance(begin,str):
            begin = Expression(begin)
            for v in begin.variables():
                if not v in self.__time_parameter_declarations:
                    if v in self.__voltage_parameter_declarations:
                        raise ValueError("Argument begin=<{}> must not refer to a voltage parameter declaration."
                                .format(str(begin)))
                    self.__time_parameter_declarations[v] = ParameterDeclaration(v)
        if isinstance(end,str):
            end = Expression(end)
            for v in end.variables():
                if not v in self.__time_parameter_declarations:
                    if v in self.__voltage_parameter_declarations:
                        raise ValueError("Argument begin=<{}> must not refer to a voltage parameter declaration."
                                .format(str(end)))
                    self.__time_parameter_declarations[v] = ParameterDeclaration(v)
        if name in self.__measurement_windows:
            self.__measurement_windows[name].append((begin,end))
        else:
            self.__measurement_windows[name] = [(begin,end)]

    @property
    def is_interruptable(self) -> bool:
        return False

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return set(self.__entries.keys())

    @property
    def num_channels(self) -> int:
        return len(self.__entries)

    def get_entries_instantiated(self, parameters: Dict[str, Parameter]) \
            -> Dict[ChannelID, List[Tuple[float, float]]]:
        """Compute an instantiated list of the table's entries.

        Args:
            parameters (Dict(str -> Parameter)): A mapping of parameter names to Parameter objects.
        Returns:
             (float, float)-list of all table entries with concrete values provided by the given
                parameters.
        """
        instantiated_entries = dict() # type: Dict[ChannelID,List[Tuple[float, float]]]
        max_time = 0

        for channel, channel_entries in self.__entries.items():
            instantiated = []
            if not channel_entries:
                instantiated.append(TableEntry(0, 0, self.__interpolation_strategies['hold']))
            else:
                for entry in channel_entries:
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
            max_time = max(max_time, instantiated[-1].t)

            # ensure that no time value occurs twice
            previous_time = -1
            for (time, _, _) in instantiated:
                if time <= previous_time:
                    raise Exception("Time value {0} is smaller than the previous value {1}."
                                    .format(time, previous_time))
                previous_time = time

            instantiated_entries[channel] = instantiated

        # ensure that all channels have equal duration
        for channel, instantiated in instantiated_entries.items():
            final_entry = instantiated[-1]
            if final_entry.t != max_time:
                instantiated.append(TableEntry(max_time,
                                               final_entry.v,
                                               self.__interpolation_strategies['hold']))
            instantiated_entries[channel] = TablePulseTemplate.__clean_entries(instantiated)
        return instantiated_entries

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

    def build_waveform(self,
                       parameters: Dict[str, Parameter],
                       measurement_mapping: Dict[str, str],
                       channel_mapping: Dict[ChannelID, ChannelID]) -> Optional['Waveform']:
        instantiated = [(channel_mapping[channel], instantiated_channel)
                        for channel, instantiated_channel in self.get_entries_instantiated(parameters).items()]
        measurement_windows = self.get_measurement_windows(parameters=parameters,
                                                           measurement_mapping=measurement_mapping)

        if len(instantiated) == 1:
            return TableWaveform(*instantiated.pop(), measurement_windows)
        else:
            return MultiChannelWaveform(
                [TableWaveform(*instantiated.pop(), measurement_windows)]
                +
                [TableWaveform(channel, instantiated_channel, [])for channel, instantiated_channel in instantiated])

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
        data['time_parameter_declarations'] = \
            [serializer.dictify(self.__time_parameter_declarations[key])
             for key in sorted(self.__time_parameter_declarations.keys())]
        data['voltage_parameter_declarations'] = \
            [serializer.dictify(self.__voltage_parameter_declarations[key])
             for key in sorted(self.__voltage_parameter_declarations.keys())]

        serialized_entries = dict()
        for channel, channel_entries in self.__entries.items():
            serialized_channel_entries = []
            for (time, voltage, interpolation) in channel_entries:
                if isinstance(time, ParameterDeclaration):
                    time = time.name
                if isinstance(voltage, ParameterDeclaration):
                    voltage = voltage.name
                serialized_channel_entries.append((time, voltage, str(interpolation)))
            serialized_entries[channel] = serialized_channel_entries
        data['entries'] = serialized_entries
        data['measurement_declarations'] = self.measurement_declarations
        data['type'] = serializer.get_type_identifier(self)
        return data

    @staticmethod
    def deserialize(serializer: Serializer,
                    time_parameter_declarations: Iterable[Any],
                    voltage_parameter_declarations: Iterable[Any],
                    entries: Dict[ChannelID,Any],
                    measurement_declarations: Dict[str,Iterable[Any]],
                    identifier: Optional[str]=None) -> 'TablePulseTemplate':
        time_parameter_declarations = \
            {declaration['name']: serializer.deserialize(declaration)
             for declaration in time_parameter_declarations}
        voltage_parameter_declarations = \
            {declaration['name']: serializer.deserialize(declaration)
             for declaration in voltage_parameter_declarations}

        template = TablePulseTemplate(channels=list(entries.keys()),
                                      identifier=identifier)

        for channel, channel_entries in entries.items():
            for (time, voltage, interpolation) in channel_entries:
                if isinstance(time, str):
                    time = time_parameter_declarations[time]
                if isinstance(voltage, str):
                    voltage = voltage_parameter_declarations[voltage]
                template.add_entry(time, voltage, interpolation=interpolation, channel=channel)

        for name, windows in measurement_declarations.items():
            for window in windows:
                template.add_measurement_declaration(name,*window)

        return template
