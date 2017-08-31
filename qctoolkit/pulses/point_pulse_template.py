from typing import Optional, List, Union, Set, Dict, Sequence
from numbers import Real
import itertools
import numbers

import numpy as np

from qctoolkit.utils.types import ChannelID
from qctoolkit.expressions import Expression
from qctoolkit.pulses.conditions import Condition
from qctoolkit.pulses.instructions import Waveform
from qctoolkit.pulses.parameters import Parameter, ParameterNotProvidedException, ParameterConstraint,\
    ParameterConstrainer
from qctoolkit.pulses.pulse_template import AtomicPulseTemplate, MeasurementDeclaration
from qctoolkit.pulses.table_pulse_template import TableEntry, EntryInInit, TableWaveform, TableWaveformEntry
from qctoolkit.pulses.measurement import MeasurementDefiner
from qctoolkit.pulses.multi_channel_pulse_template import MultiChannelWaveform


__all__ = ["PointWaveform", "PointPulseTemplate", "PointPulseEntry", "PointWaveformEntry", "InvalidPointDimension"]


PointWaveform = TableWaveform
PointWaveformEntry = TableWaveformEntry


class PointPulseEntry(TableEntry):
    def instantiate(self, parameters: Dict[str, numbers.Real], num_channels: int) -> Sequence[PointWaveformEntry]:
        t = self.t.evaluate_numeric(**parameters)
        vs = self.v.evaluate_numeric(**parameters)

        if isinstance(vs, numbers.Number):
            vs = np.full(num_channels, vs, dtype=type(vs))
        elif len(vs) != num_channels:
            raise InvalidPointDimension(expected=num_channels, received=len(vs))

        return tuple(PointWaveformEntry(t, v, self.interp)
                     for v in vs)


class PointPulseTemplate(ParameterConstrainer, MeasurementDefiner, AtomicPulseTemplate):
    def __init__(self,
                 time_point_tuple_list: List[EntryInInit],
                 channel_names: Sequence[ChannelID],
                 *,
                 parameter_constraints: Optional[List[Union[str, ParameterConstraint]]]=None,
                 measurements: Optional[List[MeasurementDeclaration]]=None,
                 identifier=None):

        AtomicPulseTemplate.__init__(self, identifier=identifier)
        ParameterConstrainer.__init__(self, parameter_constraints=parameter_constraints)
        MeasurementDefiner.__init__(self, measurements=measurements)

        self._channels = tuple(channel_names)
        self._entries = [PointPulseEntry(*tpt)
                         for tpt in time_point_tuple_list]

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return set(self._channels)

    def build_waveform(self,
                       parameters: Dict[str, Real],
                       measurement_mapping: Dict[str, str],
                       channel_mapping: Dict[ChannelID, ChannelID]) -> Optional[Waveform]:
        self.validate_parameter_constraints(parameters)

        if self.duration.evaluate_numeric(**parameters) == 0:
            return None

        mapped_channels = tuple(channel_mapping[c] for c in self._channels)

        waveform_entries = [[]]*len(mapped_channels)
        for entry in self._entries:
            for ch_entries, wf_entry in zip(waveform_entries, entry.instantiate(parameters, len(self._channels))):
                ch_entries.append(wf_entry)

        if waveform_entries[0][0].t > 0:
            for ch_entries in waveform_entries:
                ch_entries[:0] = [PointWaveformEntry(0, ch_entries[0].v, ch_entries[0].interp)]

        measurements = self.get_measurement_windows(parameters=parameters, measurement_mapping=measurement_mapping)
        if len(waveform_entries) == 1:
            return PointWaveform(mapped_channels[0], waveform_entries[0], measurement_windows=measurements)
        else:
            return MultiChannelWaveform(
                [PointWaveform(mapped_channels[0], waveform_entries[0], measurement_windows=measurements)]
                +
                [PointWaveform(channel, ch_entries, [])
                 for channel, ch_entries in zip(mapped_channels[1:], waveform_entries[1:])])

    @property
    def point_pulse_entries(self) -> Sequence[PointPulseEntry]:
        return self._entries

    def get_serialization_data(self, serializer) -> Dict:
        data = {'time_point_tuple_list': [(t.get_most_simple_representation(),
                                           v.get_most_simple_representation(),
                                           str(interp))
                                          for t, v, interp in self._entries],
                'channel_names':       self._channels}
        if self.parameter_constraints:
            data['parameter_constraints'] = [str(c) for c in self.parameter_constraints]
        if self.measurement_declarations:
            data['measurements'] = self.measurement_declarations
        return data

    @staticmethod
    def deserialize(serializer, **kwargs) -> 'PointPulseTemplate':
        return PointPulseTemplate(**kwargs)

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

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, Condition]) -> bool:
        try:
            return any(
                parameters[name].requires_stop
                for name in self.parameter_names
            )
        except KeyError as key_error:
            raise ParameterNotProvidedException(str(key_error)) from key_error


class InvalidPointDimension(Exception):
    def __init__(self, expected, received):
        super().__init__('Expected a point of dimension {} but received {}'.format(expected, received))
        self.expected = expected
        self.received = received
