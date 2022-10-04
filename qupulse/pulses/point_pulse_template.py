from typing import Optional, List, Union, Set, Dict, Sequence, Any, Tuple
from numbers import Real
import itertools
import numbers

import sympy
import numpy as np

from qupulse.utils.sympy import IndexedBroadcast
from qupulse.utils.types import ChannelID
from qupulse.expressions import Expression, ExpressionScalar
from qupulse._program.waveforms import TableWaveform, TableWaveformEntry
from qupulse.pulses.parameters import ParameterConstraint, ParameterConstrainer
from qupulse.pulses.pulse_template import AtomicPulseTemplate, MeasurementDeclaration
from qupulse.pulses.table_pulse_template import TableEntry, EntryInInit
from qupulse.pulses.multi_channel_pulse_template import MultiChannelWaveform
from qupulse.serialization import Serializer, PulseRegistryType


__all__ = ["PointWaveform", "PointPulseTemplate", "PointPulseEntry", "PointWaveformEntry", "InvalidPointDimension"]


PointWaveform = TableWaveform
PointWaveformEntry = TableWaveformEntry


class PointPulseEntry(TableEntry):
    def instantiate(self, parameters: Dict[str, numbers.Real], num_channels: int) -> Sequence[PointWaveformEntry]:
        t = self.t.evaluate_numeric(**parameters)
        vs = self.v.evaluate_numeric(**parameters)

        if isinstance(vs, numbers.Number):
            vs = (vs,) * num_channels
        elif len(vs) != num_channels:
            raise InvalidPointDimension(expected=num_channels, received=len(vs))

        return tuple(PointWaveformEntry(t, v, self.interp)
                     for v in vs)


class PointPulseTemplate(AtomicPulseTemplate, ParameterConstrainer):
    def __init__(self,
                 time_point_tuple_list: List[EntryInInit],
                 channel_names: Sequence[ChannelID],
                 *,
                 parameter_constraints: Optional[List[Union[str, ParameterConstraint]]]=None,
                 measurements: Optional[List[MeasurementDeclaration]]=None,
                 identifier: Optional[str]=None,
                 registry: PulseRegistryType=None) -> None:

        AtomicPulseTemplate.__init__(self, identifier=identifier, measurements=measurements)
        ParameterConstrainer.__init__(self, parameter_constraints=parameter_constraints)

        self._channels = tuple(channel_names)
        self._entries = [PointPulseEntry(*tpt)
                         for tpt in time_point_tuple_list]

        self._register(registry=registry)

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return set(self._channels)

    def build_waveform(self,
                       parameters: Dict[str, Real],
                       channel_mapping: Dict[ChannelID, Optional[ChannelID]]) -> Optional[Union[TableWaveform,
                                                                                                MultiChannelWaveform]]:
        self.validate_parameter_constraints(parameters=parameters, volatile=set())

        if all(channel_mapping[channel] is None
               for channel in self.defined_channels):
            return None

        if self.duration.evaluate_numeric(**parameters) == 0:
            return None

        mapped_channels = tuple(channel_mapping[c] for c in self._channels)

        waveform_entries = list([] for _ in range(len(self._channels)))
        for entry in self._entries:
            instantiated_entries = entry.instantiate(parameters, len(self._channels))
            for ch_entries, wf_entry in zip(waveform_entries, instantiated_entries):
                ch_entries.append(wf_entry)

        if waveform_entries[0][0].t > 0:
            for ch_entries in waveform_entries:
                ch_entries[:0] = [PointWaveformEntry(0, ch_entries[0].v, ch_entries[0].interp)]

        # filter mappings to None
        channel_entries = [(ch, ch_entries)
                           for (ch, ch_entries) in zip(mapped_channels, waveform_entries)
                           if ch is not None]
        mapped_channels, waveform_entries = zip(*channel_entries)

        waveforms = [PointWaveform.from_table(mapped_channel, ch_entries)
                     for mapped_channel, ch_entries in zip(mapped_channels, waveform_entries)]

        return MultiChannelWaveform.from_parallel(waveforms)

    @property
    def point_pulse_entries(self) -> Sequence[PointPulseEntry]:
        return self._entries

    def get_serialization_data(self, serializer: Optional[Serializer]=None) -> Dict[str, Any]:
        data = super().get_serialization_data(serializer)

        if serializer: # compatibility to old serialization routines, deprecated
            data = dict()

        data['time_point_tuple_list'] = [entry.get_serialization_data() for entry in self._entries]
        data['channel_names'] = self._channels
        if self.parameter_constraints:
            data['parameter_constraints'] = [str(c) for c in self.parameter_constraints]
        if self.measurement_declarations:
            data['measurements'] = self.measurement_declarations
        return data

    @property
    def duration(self) -> Expression:
        return self._entries[-1].t

    @property
    def point_parameters(self) -> Set[str]:
        return set(
            var
            for time, point, *_ in self._entries
            for var in itertools.chain(time.variables, point.variables)
        )

    @property
    def parameter_names(self) -> Set[str]:
        return self.point_parameters | self.measurement_parameters | self.constrained_parameters

    @property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        expressions = {}
        shape = (len(self.defined_channels),)

        for i, channel in enumerate(self._channels):
            def value_trafo(v):
                try:
                    return v.underlying_expression[i]
                except TypeError:
                    return IndexedBroadcast(v.underlying_expression, shape, i)
            pre_entry = TableEntry(0, self._entries[0].v, None)
            entries = [pre_entry] + self._entries
            expressions[channel] = TableEntry._sequence_integral(entries, expression_extractor=value_trafo)
        return expressions

    def _as_expression(self) -> Dict[ChannelID, ExpressionScalar]:
        t = self._AS_EXPRESSION_TIME
        shape = (len(self.defined_channels),)
        expressions = {}

        for i, channel in enumerate(self._channels):
            def value_trafo(v):
                try:
                    return v.underlying_expression[i]
                except TypeError:
                    return IndexedBroadcast(v.underlying_expression, shape, i)
            pre_value = value_trafo(self._entries[0].v)
            post_value = value_trafo(self._entries[-1].v)
            pw = TableEntry._sequence_as_expression(self._entries,
                                                    expression_extractor=value_trafo,
                                                    t=t,
                                                    post_value=post_value,
                                                    pre_value=pre_value)
            expressions[channel] = pw
        return expressions

    @property
    def initial_values(self) -> Dict[ChannelID, ExpressionScalar]:
        shape = (len(self._channels),)
        return {
            ch: ExpressionScalar(IndexedBroadcast(self._entries[0].v, shape, ch_idx))
            for ch_idx, ch in enumerate(self._channels)
        }

    @property
    def final_values(self) -> Dict[ChannelID, ExpressionScalar]:
        shape = (len(self._channels),)
        return {
            ch: ExpressionScalar(IndexedBroadcast(self._entries[-1].v, shape, ch_idx))
            for ch_idx, ch in enumerate(self._channels)
        }


class InvalidPointDimension(Exception):
    def __init__(self, expected, received):
        super().__init__('Expected a point of dimension {} but received {}'.format(expected, received))
        self.expected = expected
        self.received = received
