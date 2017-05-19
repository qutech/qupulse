"""This module defines the TablePulseTemplate, one of the elementary pulse templates and its
waveform representation.

Classes:
    - TablePulseTemplate: Defines a pulse via interpolation of a sequence of (time,voltage)-pairs.
    - TableWaveform: A waveform instantiated from a TablePulseTemplate by providing values for its
        declared parameters.
"""

from typing import Union, Dict, List, Set, Optional, NamedTuple, Any, Iterable, Tuple
import numbers
import itertools
import warnings

import numpy as np
import sympy

from qctoolkit import MeasurementWindow, ChannelID
from qctoolkit.serialization import Serializer
from qctoolkit.pulses.parameters import Parameter, \
    ParameterNotProvidedException, ParameterConstraint, ParameterConstraintViolation, ParameterConstrainer
from qctoolkit.pulses.pulse_template import AtomicPulseTemplate, MeasurementDeclaration
from qctoolkit.pulses.interpolation import InterpolationStrategy, LinearInterpolationStrategy, \
    HoldInterpolationStrategy, JumpInterpolationStrategy
from qctoolkit.pulses.instructions import Waveform
from qctoolkit.pulses.conditions import Condition
from qctoolkit.expressions import Expression
from qctoolkit.pulses.multi_channel_pulse_template import MultiChannelWaveform
from qctoolkit.pulses.measurement import MeasurementDefiner

__all__ = ["TablePulseTemplate", "TableWaveform", "WaveformTableEntry"]


WaveformTableEntry = NamedTuple(
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
        super().__init__()
        self._table = tuple(waveform_table)
        self._channel_id = channel
        self._measurement_windows = tuple(measurement_windows)

        if len(waveform_table) < 2:
            raise ValueError("A given waveform table has less than two entries.")

    @property
    def compare_key(self) -> Any:
        return self._channel_id, self._table, self._measurement_windows

    @property
    def duration(self) -> float:
        return self._table[-1].t

    def unsafe_sample(self,
                      channel: ChannelID,
                      sample_times: np.ndarray,
                      output_array: Union[np.ndarray, None]=None) -> np.ndarray:
        if output_array is None:
            output_array = np.empty(len(sample_times))

        for entry1, entry2 in zip(self._table[:-1], self._table[1:]):
            indices = slice(np.searchsorted(sample_times, entry1.t, 'left'),
                            np.searchsorted(sample_times, entry2.t, 'right'))
            output_array[indices] = \
                entry2.interp((entry1.t, entry1.v), (entry2.t, entry2.v), sample_times[indices])
        return output_array

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return {self._channel_id}

    def get_measurement_windows(self) -> Iterable[MeasurementWindow]:
        return self._measurement_windows

    def unsafe_get_subset_for_channels(self, channels: Set[ChannelID]) -> 'Waveform':
        return self

ValueInInit = Union[Expression, str, numbers.Real]
EntryInInit = Union['TableEntry',
                    Tuple[ValueInInit, ValueInInit],
                    Tuple[ValueInInit, ValueInInit, Union[str, InterpolationStrategy]]]


class TableEntry(tuple):
    def __new__(cls, t: ValueInInit, v: ValueInInit, interp: Union[str, InterpolationStrategy]='hold'):
        return tuple.__new__(cls, (t if isinstance(t, Expression) else Expression(t),
                                   v if isinstance(v, Expression) else Expression(v),
                                   interp if isinstance(interp, InterpolationStrategy)
                                   else TablePulseTemplate.interpolation_strategies[interp]))

    @property
    def t(self) -> Expression:
        return self[0]

    @property
    def v(self) -> Expression:
        return self[1]

    @property
    def interp(self) -> InterpolationStrategy:
        return self[2]


class TablePulseTemplate(AtomicPulseTemplate, ParameterConstrainer, MeasurementDefiner):
    interpolation_strategies = {'linear': LinearInterpolationStrategy(),
                                'hold': HoldInterpolationStrategy(),
                                'jump': JumpInterpolationStrategy()}

    def __init__(self, entries: Dict[ChannelID, List[EntryInInit]],
                 identifier: Optional[str]=None,
                 *,
                 parameter_constraints: Optional[List[Union[str, ParameterConstraint]]]=None,
                 measurements: Optional[List[MeasurementDeclaration]]=None,
                 consistency_check=True):
        AtomicPulseTemplate.__init__(self, identifier=identifier)
        ParameterConstrainer.__init__(self, parameter_constraints=parameter_constraints)
        MeasurementDefiner.__init__(self, measurements=measurements)

        self._entries = dict((ch, list()) for ch in entries.keys())
        for channel, channel_entries in entries.items():
            if len(channel_entries) == 0:
                raise ValueError('Channel {} is empty'.format(channel))

            for entry in channel_entries:
                self._add_entry(channel, TableEntry(*entry))

        if self.duration == 0:
            warnings.warn('Table pulse template with duration 0 on construction.',
                          category=ZeroDurationTablePulseTemplate)

        if consistency_check:
            # perform a simple consistency check. All inequalities with more than one free variable are ignored as the
            # sympy solver does not support them

            # collect all conditions
            inequalities = [eq.sympified_expression for eq in self._parameter_constraints] +\
                           [sympy.Le(previous_entry.t.compare_key, entry.t.compare_key)
                            for channel_entries in self._entries.values()
                            for previous_entry, entry in zip(channel_entries, channel_entries[1:])]

            # test if any condition is already dissatisfied
            if any(isinstance(eq, sympy.boolalg.BooleanAtom) and bool(eq) is False
                   for eq in inequalities):
                raise ValueError('Table pulse template has impossible parametrization')

            # filter conditions that are inequalities with one free variable and test if the solution set is empty
            inequalities = [eq for eq in inequalities if isinstance(eq, sympy.Rel) and len(eq.free_symbols) == 1]
            if not sympy.reduce_inequalities(inequalities):
                raise ValueError('Table pulse template has impossible parametrization')

    def _add_entry(self, channel, new_entry: TableEntry) -> None:

        # comparisons with Expression can yield None -> use 'is True' and 'is False'
        if (new_entry.t < 0) is True:
            raise ValueError('Time parameter number {} of channel {} is negative.'.format(
                len(self._entries[channel]), channel))

        for previous_entry in self._entries[channel]:
            if (new_entry.t < previous_entry.t) is True:
                raise ValueError('Time parameter number {} of channel {} is smaller than a previous one'.format(
                    len(self._entries[channel]), channel))

        self._entries[channel].append(new_entry)

    @property
    def entries(self) -> Dict[ChannelID, List[TableEntry]]:
        return self._entries

    @property
    def measurement_names(self) -> Set[str]:
        return {name for name, _, _ in self._measurement_windows}

    def get_entries_instantiated(self, parameters: Dict[str, numbers.Real]) \
            -> Dict[ChannelID, List[WaveformTableEntry]]:
        """Compute an instantiated list of the table's entries.

        Args:
            parameters (Dict(str -> Parameter)): A mapping of parameter names to Parameter objects.
        Returns:
             (float, float)-list of all table entries with concrete values provided by the given
                parameters.
        """
        if not (self.table_parameters <= set(parameters.keys())):
            raise ParameterNotProvidedException((self.table_parameters - set(parameters.keys())).pop())

        instantiated_entries = dict()  # type: Dict[ChannelID,List[WaveformTableEntry]]

        for channel, channel_entries in self._entries.items():
            instantiated = [WaveformTableEntry(entry.t.evaluate_numeric(**parameters),
                                               entry.v.evaluate_numeric(**parameters),
                                               entry.interp)
                            for entry in channel_entries]

            # Add (0, v) entry if wf starts at finite time
            if instantiated[0].t > 0:
                instantiated.insert(0, WaveformTableEntry(0,
                                                          instantiated[0].v,
                                                          TablePulseTemplate.interpolation_strategies['hold']))

            for (previous_time, _, _), (time, _, _) in zip(instantiated, instantiated[1:]):
                if time < previous_time:
                    raise Exception("Time value {0} is smaller than the previous value {1}."
                                    .format(time, previous_time))
            instantiated_entries[channel] = instantiated

        duration = max(instantiated[-1].t for instantiated in instantiated_entries.values())

        # ensure that all channels have equal duration
        for channel, instantiated in instantiated_entries.items():
            final_entry = instantiated[-1]
            if final_entry.t < duration:
                instantiated.append(WaveformTableEntry(duration,
                                                       final_entry.v,
                                                       TablePulseTemplate.interpolation_strategies['hold']))
            instantiated_entries[channel] = TablePulseTemplate._remove_redundant_entries(instantiated)
        return instantiated_entries

    @staticmethod
    def _remove_redundant_entries(entries: List[WaveformTableEntry]) -> List[WaveformTableEntry]:
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

        for index in range(length - 2, 0, -1):
            previous_step = entries[index - 1]
            step = entries[index]
            next_step = entries[index + 1]
            if step.v == previous_step.v and step.v == next_step.v:
                entries.pop(index)
        return entries

    @property
    def table_parameters(self) -> Set[str]:
        return set(
            var
            for channel_entries in self.entries.values()
            for entry in channel_entries
            for var in itertools.chain(entry.t.variables, entry.v.variables)
        ) | self.constrained_parameters

    @property
    def parameter_names(self) -> Set[str]:
        return self.table_parameters | self.measurement_parameters

    @property
    def is_interruptable(self) -> bool:
        return False

    @property
    def duration(self) -> Expression:
        return Expression('Max({})'.format(','.join(
            (str(entries[-1].t) for entries in self._entries.values())
        )))

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return set(self._entries.keys())

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

    @property
    def num_channels(self) -> int:
        return len(self._entries)

    def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
        return dict(
            entries=dict(
                (channel, [(entry.t.get_most_simple_representation(),
                            entry.v.get_most_simple_representation(),
                            str(entry.interp)) for entry in channel_entries])
                for channel, channel_entries in self._entries.items()
            ),
            parameter_constraints=[str(c) for c in self.parameter_constraints],
            measurements=self.measurement_declarations
        )

    @staticmethod
    def deserialize(serializer: Serializer,
                    entries: Dict[ChannelID, List[EntryInInit]],
                    parameter_constraints: List[str],
                    measurements: List[MeasurementDeclaration],
                    identifier: Optional[str]=None) -> 'TablePulseTemplate':
        return TablePulseTemplate(entries=entries,
                                  identifier=identifier,
                                  parameter_constraints=parameter_constraints,
                                  measurements=measurements,
                                  consistency_check=False)

    def build_waveform(self,
                       parameters: Dict[str, numbers.Real],
                       measurement_mapping: Dict[str, str],
                       channel_mapping: Dict[ChannelID, ChannelID]) -> Optional['Waveform']:
        self.validate_parameter_constraints(parameters)

        instantiated = [(channel_mapping[channel], instantiated_channel)
                        for channel, instantiated_channel in self.get_entries_instantiated(parameters).items()]
        if self.duration.evaluate_numeric(**parameters) == 0:
            return None

        measurements = self.get_measurement_windows(parameters=parameters, measurement_mapping=measurement_mapping)
        if len(instantiated) == 1:
            return TableWaveform(*instantiated.pop(), measurement_windows=measurements)
        else:
            return MultiChannelWaveform(
                [TableWaveform(*instantiated.pop(), measurement_windows=measurements)]
                +
                [TableWaveform(channel, instantiated_channel, [])for channel, instantiated_channel in instantiated])

    @staticmethod
    def from_array(times: np.ndarray, voltages: np.ndarray, channels: List[ChannelID]) -> 'TablePulseTemplate':
        """Static constructor to build a TablePulse from numpy arrays.

        Args:
            times: 1D numpy array with time values
            voltages: 1D or 2D numpy array with voltage values
            channels: channels to define

        Returns:
            TablePulseTemplate with the given values, hold interpolation everywhere and no free
            parameters.
        """
        if times.ndim == 0 or voltages.ndim == 0:
            raise ValueError('Zero dimensional input is not accepted.')

        if times.ndim > 2 or voltages.ndim > 2:
            raise ValueError('Three or higher dimensional input is not accepted.')

        if times.ndim == 2 and times.shape[0] != len(channels):
            raise ValueError('First dimension of times must be equal to the number of channels')

        if voltages.ndim == 2 and voltages.shape[0] != len(channels):
            raise ValueError('First dimension of voltages must be equal to the number of channels')

        if voltages.shape[-1] != times.shape[-1]:
            ValueError('Different number of entries for times and voltages')

        return TablePulseTemplate(dict((channel, list(zip(times if times.ndim == 1 else times[i, :],
                                                          voltages if voltages.ndim == 1 else voltages[i, :])))
                                       for i, channel in enumerate(channels)))


class ZeroDurationTablePulseTemplate(UserWarning):
    pass
