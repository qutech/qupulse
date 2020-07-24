"""This module defines the TablePulseTemplate, one of the elementary pulse templates and its
waveform representation.

Classes:
    - TablePulseTemplate: Defines a pulse via interpolation of a sequence of (time,voltage)-pairs.
    - TableWaveform: A waveform instantiated from a TablePulseTemplate by providing values for its
        declared parameters.
"""

from typing import Union, Dict, List, Set, Optional, Any, Tuple, Sequence, NamedTuple
import numbers
import itertools
import warnings

import numpy as np
import sympy
from sympy.logic.boolalg import BooleanAtom

from qupulse.utils.types import ChannelID
from qupulse.serialization import Serializer, PulseRegistryType
from qupulse.pulses.parameters import Parameter, \
    ParameterNotProvidedException, ParameterConstraint, ParameterConstrainer
from qupulse.pulses.pulse_template import AtomicPulseTemplate, MeasurementDeclaration
from qupulse.pulses.interpolation import InterpolationStrategy, LinearInterpolationStrategy, \
    HoldInterpolationStrategy, JumpInterpolationStrategy
from qupulse._program.waveforms import TableWaveform, TableWaveformEntry
from qupulse.expressions import ExpressionScalar, Expression
from qupulse.pulses.multi_channel_pulse_template import MultiChannelWaveform

__all__ = ["TablePulseTemplate", "concatenate"]


ValueInInit = Union[ExpressionScalar, str, numbers.Real]
EntryInInit = Union['TableEntry',
                    Tuple[ValueInInit, ValueInInit],
                    Tuple[ValueInInit, ValueInInit, Union[str, InterpolationStrategy]]]


class TableEntry(NamedTuple('TableEntry', [('t', ExpressionScalar),
                                           ('v', Expression),
                                           ('interp', InterpolationStrategy)])):
    __slots__ = ()

    def __new__(cls, t: ValueInInit, v: ValueInInit, interp: Union[str, InterpolationStrategy]='default'):
        if interp in TablePulseTemplate.interpolation_strategies:
            interp = TablePulseTemplate.interpolation_strategies[interp]
        if not isinstance(interp, InterpolationStrategy):
            raise KeyError(interp, 'is not a valid interpolation strategy')

        return super().__new__(cls, ExpressionScalar.make(t),
                                    Expression.make(v),
                                    interp)

    def instantiate(self, parameters: Dict[str, numbers.Real]) -> TableWaveformEntry:
        return TableWaveformEntry(self.t.evaluate_numeric(**parameters),
                                  self.v.evaluate_numeric(**parameters),
                                  self.interp)

    def get_serialization_data(self) -> tuple:
        return self.t.get_serialization_data(), self.v.get_serialization_data(), str(self.interp)


class TablePulseTemplate(AtomicPulseTemplate, ParameterConstrainer):
    """The TablePulseTemplate class implements pulses described by a table with time, voltage and interpolation strategy
    inputs. The interpolation strategy describes how the voltage between the entries is interpolated(see also
    InterpolationStrategy.) It can define multiple channels of which each has a separate table. If they do not have the
    same length the shorter channels are extended to the longest duration.

    If the time entries of all channels are equal it is more convenient to use the :paramrefPointPulseTemplate`."""
    interpolation_strategies = {'linear': LinearInterpolationStrategy(),
                                'hold': HoldInterpolationStrategy(),
                                'jump': JumpInterpolationStrategy(),
                                'default': HoldInterpolationStrategy()}

    def __init__(self, entries: Dict[ChannelID, Sequence[EntryInInit]],
                 identifier: Optional[str]=None,
                 *,
                 parameter_constraints: Optional[List[Union[str, ParameterConstraint]]]=None,
                 measurements: Optional[List[MeasurementDeclaration]]=None,
                 consistency_check: bool=True,
                 registry: PulseRegistryType=None) -> None:
        """
        Construct a `TablePulseTemplate` from a dict which maps channels to their entries. By default the consistency
        of the provided entries is checked. There are two static functions for convenience construction: from_array and
        from_entry_list.

        Args:
            entries: A dictionary that maps channel ids to a list of entries. An entry is a
                (time, voltage[, interpolation strategy]) tuple or a TableEntry
            identifier: Used for serialization
            parameter_constraints: Constraint list that is forwarded to the ParameterConstrainer superclass
            measurements: Measurement declaration list that is forwarded to the MeasurementDefiner superclass
            consistency_check: If True the consistency of the times will be checked on construction as far as possible
        """
        AtomicPulseTemplate.__init__(self, identifier=identifier, measurements=measurements)
        ParameterConstrainer.__init__(self, parameter_constraints=parameter_constraints)

        if not entries:
            raise ValueError("Cannot construct an empty TablePulseTemplate (no entries given). There is currently no "
                             "specific reason for this. Please submit an issue if you need this 'feature'.")

        self._entries = dict((ch, list()) for ch in entries.keys())
        for channel, channel_entries in entries.items():
            if len(channel_entries) == 0:
                raise ValueError('Channel {} is empty'.format(channel))

            for entry in channel_entries:
                self._add_entry(channel, TableEntry(*entry))

        self._duration = self.calculate_duration()
        self._table_parameters = set(
            var
            for channel_entries in self.entries.values()
            for entry in channel_entries
            for var in itertools.chain(entry.t.variables, entry.v.variables)
        ) | self.constrained_parameters

        if self.duration == 0:
            warnings.warn('Table pulse template with duration 0 on construction.',
                          category=ZeroDurationTablePulseTemplate)

        if consistency_check:
            # perform a simple consistency check. All inequalities with more than one free variable are ignored as the
            # sympy solver does not support them

            # collect all conditions
            inequalities = [eq.sympified_expression for eq in self._parameter_constraints] +\
                           [sympy.Le(previous_entry.t.underlying_expression, entry.t.underlying_expression)
                            for channel_entries in self._entries.values()
                            for previous_entry, entry in zip(channel_entries, channel_entries[1:])]

            # test if any condition is already dissatisfied
            if any(isinstance(eq, BooleanAtom) and bool(eq) is False
                   for eq in inequalities):
                raise ValueError('Table pulse template has impossible parametrization')

            # filter conditions that are inequalities with one free variable and test if the solution set is empty
            inequalities = [eq for eq in inequalities if isinstance(eq, sympy.Rel) and len(eq.free_symbols) == 1]
            if not sympy.reduce_inequalities(inequalities):
                raise ValueError('Table pulse template has impossible parametrization')

        self._register(registry=registry)

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

    def get_entries_instantiated(self, parameters: Dict[str, numbers.Real]) \
            -> Dict[ChannelID, List[TableWaveformEntry]]:
        """Compute an instantiated list of the table's entries.

        Args:
            parameters (Dict(str -> Parameter)): A mapping of parameter names to Parameter objects.
        Returns:
             (float, float)-list of all table entries with concrete values provided by the given
                parameters.
        """
        if not (self.table_parameters <= set(parameters.keys())):
            raise ParameterNotProvidedException((self.table_parameters - set(parameters.keys())).pop())

        instantiated_entries = dict()  # type: Dict[ChannelID,List[TableWaveformEntry]]

        for channel, channel_entries in self._entries.items():
            instantiated = [entry.instantiate(parameters)
                            for entry in channel_entries]

            # Add (0, v) entry if wf starts at finite time
            if instantiated[0].t > 0:
                instantiated.insert(0, TableWaveformEntry(0,
                                                          instantiated[0].v,
                                                          TablePulseTemplate.interpolation_strategies['hold']))
            instantiated_entries[channel] = instantiated

        duration = max(instantiated[-1].t for instantiated in instantiated_entries.values())

        # ensure that all channels have equal duration
        for channel, instantiated in instantiated_entries.items():
            final_entry = instantiated[-1]
            if final_entry.t < duration:
                instantiated.append(TableWaveformEntry(duration,
                                                       final_entry.v,
                                                       TablePulseTemplate.interpolation_strategies['hold']))
            instantiated_entries[channel] = instantiated
        return instantiated_entries

    @property
    def table_parameters(self) -> Set[str]:
        return self._table_parameters

    @property
    def parameter_names(self) -> Set[str]:
        return self.table_parameters | self.measurement_parameters | self.constrained_parameters

    @property
    def duration(self) -> ExpressionScalar:
        return self._duration

    def calculate_duration(self) -> ExpressionScalar:
        duration_expressions = [entries[-1].t for entries in self._entries.values()]
        duration_expression = sympy.Max(*(expr.sympified_expression for expr in duration_expressions))
        return ExpressionScalar(duration_expression)

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return set(self._entries.keys())

    def get_serialization_data(self, serializer: Optional[Serializer]=None) -> Dict[str, Any]:
        data = super().get_serialization_data(serializer)

        if serializer: # compatibility to old serialization routines, deprecated
            data = dict()

        local_data = dict(
            entries=dict(
                (channel, [entry.get_serialization_data()
                           for entry in channel_entries])
                for channel, channel_entries in self.entries.items()
            ),
            parameter_constraints=[str(c) for c in self.parameter_constraints],
            measurements=self.measurement_declarations
        )
        data.update(**local_data)
        return data

    def build_waveform(self,
                       parameters: Dict[str, numbers.Real],
                       channel_mapping: Dict[ChannelID, Optional[ChannelID]]) -> Optional[Union[TableWaveform,
                                                                                                MultiChannelWaveform]]:
        self.validate_parameter_constraints(parameters, volatile=set())

        if all(channel_mapping[channel] is None
               for channel in self.defined_channels):
            return None

        instantiated = [(channel_mapping[channel], instantiated_channel)
                        for channel, instantiated_channel in self.get_entries_instantiated(parameters).items()
                        if channel_mapping[channel] is not None]

        if self.duration.evaluate_numeric(**parameters) == 0:
            return None

        waveforms = [TableWaveform(*ch_instantiated)
                     for ch_instantiated in instantiated]

        if len(waveforms) == 1:
            return waveforms.pop()
        else:
            return MultiChannelWaveform(waveforms)

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

    @staticmethod
    def from_entry_list(entry_list: List[Tuple],
                        channel_names: Optional[List[ChannelID]]=None, **kwargs) -> 'TablePulseTemplate':
        """Static constructor for a TablePulseTemplate where all channel's entries share the same times.

        :param entry_list: List of tuples of the form (t, v_1, ..., v_N[, interp])
        :param channel_names: Optional list of channel identifiers to use. Default is [0, ..., N-1]
        :param kwargs: Forwarded to TablePulseTemplate constructor
        :return: TablePulseTemplate with
        """
        # TODO: Better doc string
        def is_valid_interpolation_strategy(inter):
            return inter in TablePulseTemplate.interpolation_strategies or isinstance(inter, InterpolationStrategy)

        # determine number of channels
        max_len = max(len(data) for data in entry_list)
        min_len = min(len(data) for data in entry_list)

        if max_len - min_len > 1:
            raise ValueError('There are entries of contradicting lengths: {}'.format(set(len(t) for t in entry_list)))
        elif max_len - min_len == 1:
            num_chan = min_len - 1
        else:
            # figure out whether all last entries are interpolation strategies
            if all(is_valid_interpolation_strategy(interp) for *data, interp in entry_list):
                num_chan = min_len - 2
            else:
                num_chan = min_len - 1

        # insert default interpolation strategy key
        entry_list = [(t, *data, interp) if len(data) == num_chan else (t, *data, interp, 'default')
                      for t, *data, interp in entry_list]

        for *_, last_voltage, _ in entry_list:
            if last_voltage in TablePulseTemplate.interpolation_strategies:
                warnings.warn('{} is also an interpolation strategy name but handled as a voltage. Is it intended?'
                              .format(last_voltage), AmbiguousTablePulseEntry)

        if channel_names is None:
            channel_names = list(range(num_chan))
        elif len(channel_names) != num_chan:
            raise ValueError('Number of channel identifiers does not correspond to the number of channels.')

        parsed = {channel_name: [] for channel_name in channel_names}

        for time, *voltages, interp in entry_list:
            for channel_name, volt in zip(channel_names, voltages):
                parsed[channel_name].append((time, volt, interp))

        return TablePulseTemplate(parsed, **kwargs)

    @property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        expressions = dict()
        for channel, channel_entries in self._entries.items():

            expr = 0
            for first_entry, second_entry in zip(channel_entries[:-1], channel_entries[1:]):
                substitutions = {'t0': ExpressionScalar(first_entry.t).sympified_expression, 'v0': ExpressionScalar(first_entry.v).sympified_expression,
                                 't1': ExpressionScalar(second_entry.t).sympified_expression, 'v1': ExpressionScalar(second_entry.v).sympified_expression}

                expr += first_entry.interp.integral.sympified_expression.subs(substitutions)
            expressions[channel] = ExpressionScalar(expr)

        return expressions


def concatenate(*table_pulse_templates: TablePulseTemplate, **kwargs) -> TablePulseTemplate:
    """Concatenate two or more table pulse templates"""
    first_template, *other_templates = table_pulse_templates

    entries = {channel: [] for channel in first_template.defined_channels}
    duration = ExpressionScalar(0)

    for i, template in enumerate(table_pulse_templates):
        if not isinstance(template, TablePulseTemplate):
            raise TypeError('Template number %d is not a TablePulseTemplate' % i)

        new_duration = duration + template.duration

        if template.defined_channels != first_template.defined_channels:
            raise ValueError('Template number %d has differing defined channels' % i,
                             first_template.defined_channels, template.defined_channels)

        for channel, channel_entries in template.entries.items():
            first_t, first_v, _ = channel_entries[0]
            if i > 0 and first_t != 0:
                if (first_v == 0) is False:
                    entries[channel].append((duration, first_v, 'hold'))

            for t, v, interp in channel_entries:
                entries[channel].append((duration.sympified_expression + t, v, interp))

            last_t, last_v, _ = channel_entries[-1]
            if i < len(other_templates) and last_t != new_duration:
                entries[channel].append((new_duration, last_v, TablePulseTemplate.interpolation_strategies['hold']))

        duration = new_duration

    return TablePulseTemplate(entries, **kwargs)


class ZeroDurationTablePulseTemplate(UserWarning):
    pass


class AmbiguousTablePulseEntry(UserWarning):
    pass
