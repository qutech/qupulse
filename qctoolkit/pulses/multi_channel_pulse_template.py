"""This module defines MultiChannelPulseTemplate, which allows the combination of several
AtomicPulseTemplates into a single template spanning several channels.

Classes:
    - MultiChannelPulseTemplate: A pulse template defined for several channels by combining pulse
        templates
    - MultiChannelWaveform: A waveform defined for several channels by combining waveforms
"""

from typing import Dict, List, Optional, Any, Iterable, Union, Set, Sequence
import numbers
import warnings

import numpy


from qctoolkit.serialization import Serializer, PulseRegistryType

from qctoolkit.utils.types import ChannelID, TimeType
from qctoolkit._program.waveforms import MultiChannelWaveform
from qctoolkit.pulses.pulse_template import PulseTemplate, AtomicPulseTemplate
from qctoolkit.pulses.mapping_pulse_template import MappingPulseTemplate, MappingTuple
from qctoolkit.pulses.parameters import Parameter, ParameterConstrainer
from qctoolkit.pulses.measurement import MeasurementDeclaration, MeasurementWindow
from qctoolkit.expressions import Expression, ExpressionScalar

__all__ = ["AtomicMultiChannelPulseTemplate"]


class AtomicMultiChannelPulseTemplate(AtomicPulseTemplate, ParameterConstrainer):
    def __init__(self,
                 *subtemplates: Union[AtomicPulseTemplate, MappingTuple, MappingPulseTemplate],
                 external_parameters: Optional[Set[str]]=None,
                 identifier: Optional[str]=None,
                 parameter_constraints: Optional[List]=None,
                 measurements: Optional[List[MeasurementDeclaration]]=None,
                 registry: PulseRegistryType=None) -> None:
        AtomicPulseTemplate.__init__(self, identifier=identifier, measurements=measurements)
        ParameterConstrainer.__init__(self, parameter_constraints=parameter_constraints)

        self._subtemplates = [st if isinstance(st, PulseTemplate) else MappingPulseTemplate.from_tuple(st) for st in
                              subtemplates]

        for subtemplate in self._subtemplates:
            if isinstance(subtemplate, AtomicPulseTemplate):
                continue
            elif isinstance(subtemplate, MappingPulseTemplate):
                if isinstance(subtemplate.template, AtomicPulseTemplate):
                    continue
                else:
                    raise TypeError('Non atomic subtemplate of MappingPulseTemplate: {}'.format(subtemplate.template))
            else:
                raise TypeError('Non atomic subtemplate: {}'.format(subtemplate))

        if not self._subtemplates:
            raise ValueError('Cannot create empty MultiChannelPulseTemplate')

        defined_channels = [st.defined_channels for st in self._subtemplates]

        # check there are no intersections between channels
        for i, channels_i in enumerate(defined_channels):
            for j, channels_j in enumerate(defined_channels[i + 1:]):
                if channels_i & channels_j:
                    raise ChannelMappingException('subtemplate {}'.format(i + 1),
                                                  'subtemplate {}'.format(i + 2 + j),
                                                  (channels_i | channels_j).pop())

        if external_parameters is not None:
            warnings.warn("external_parameters is an obsolete argument and will be removed in the future.",
                          category=DeprecationWarning)

        duration = self._subtemplates[0].duration
        for subtemplate in self._subtemplates[1:]:
            if (duration == subtemplate.duration) is True:
                continue
            else:
                raise ValueError('Could not assert duration equality of {} and {}'.format(duration,
                                                                                          subtemplate.duration))

        self._register(registry=registry)

    @property
    def duration(self) -> Expression:
        return self._subtemplates[0].duration

    @property
    def parameter_names(self) -> Set[str]:
        return set.union(self.measurement_parameters,
                         self.constrained_parameters,
                         *(st.parameter_names for st in self._subtemplates))

    @property
    def subtemplates(self) -> Sequence[AtomicPulseTemplate]:
        return self._subtemplates

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return set.union(*(st.defined_channels for st in self._subtemplates))

    @property
    def measurement_names(self) -> Set[str]:
        return super().measurement_names.union(*(st.measurement_names for st in self._subtemplates))

    def build_waveform(self, parameters: Dict[str, numbers.Real],
                       channel_mapping: Dict[ChannelID, Optional[ChannelID]]) -> Optional['MultiChannelWaveform']:
        self.validate_parameter_constraints(parameters=parameters)

        sub_waveforms = []
        for subtemplate in self.subtemplates:
            sub_waveform = subtemplate.build_waveform(parameters,
                                                      channel_mapping=channel_mapping)
            if sub_waveform is not None:
                sub_waveforms.append(sub_waveform)

        if len(sub_waveforms) == 0:
            return None
        if len(sub_waveforms) == 1:
            return sub_waveforms[0]
        else:
            return MultiChannelWaveform(sub_waveforms)

    def get_measurement_windows(self,
                                parameters: Dict[str, numbers.Real],
                                measurement_mapping: Dict[str, Optional[str]]) -> List[MeasurementWindow]:
        measurements = super().get_measurement_windows(parameters=parameters,
                                                       measurement_mapping=measurement_mapping)
        for st in self.subtemplates:
            measurements.extend(st.get_measurement_windows(parameters=parameters,
                                                           measurement_mapping=measurement_mapping))
        return measurements

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, 'Condition']) -> bool:
        return any(st.requires_stop(parameters, conditions) for st in self._subtemplates)

    def get_serialization_data(self) -> Dict[str, Any]:
        data = super().get_serialization_data()
        data['subtemplates'] = self.subtemplates

        if self.parameter_constraints:
            data['parameter_constraints'] = [str(constraint) for constraint in self.parameter_constraints]
        if self.measurement_declarations:
            data['measurements'] = self.measurement_declarations

        return data

    @classmethod
    def deserialize(cls, serializer: Optional[Serializer]=None, **kwargs) -> 'AtomicMultiChannelPulseTemplate':
        subtemplates = kwargs['subtemplates']
        del kwargs['subtemplates']

        if serializer: # compatibility to old serialization routines, deprecated
            subtemplates = [serializer.deserialize(st) for st in subtemplates]

        return cls(*subtemplates, **kwargs)

    @property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        expressions = dict()
        for subtemplate in self._subtemplates:
            expressions.update(subtemplate.integral)
        return expressions


class ChannelMappingException(Exception):
    def __init__(self, obj1, obj2, intersect_set):
        self.intersect_set = intersect_set
        self.obj1 = obj1
        self.obj2 = obj2

    def __str__(self) -> str:
        return 'Channel <{chs}> is defined in {o1} and {o2}'.format(chs=self.intersect_set, o1=self.obj1, o2=self.obj2)