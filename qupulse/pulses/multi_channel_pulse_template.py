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

from qupulse.serialization import Serializer, PulseRegistryType

from qupulse.pulses.conditions import Condition
from qupulse.utils import isclose
from qupulse.utils.sympy import almost_equal
from qupulse.utils.types import ChannelID, TimeType
from qupulse._program.waveforms import MultiChannelWaveform, Waveform
from qupulse.pulses.pulse_template import PulseTemplate, AtomicPulseTemplate
from qupulse.pulses.mapping_pulse_template import MappingPulseTemplate, MappingTuple
from qupulse.pulses.parameters import Parameter, ParameterConstrainer
from qupulse.pulses.measurement import MeasurementDeclaration, MeasurementWindow
from qupulse.expressions import Expression, ExpressionScalar

__all__ = ["AtomicMultiChannelPulseTemplate"]


class AtomicMultiChannelPulseTemplate(AtomicPulseTemplate, ParameterConstrainer):
    """Combines multiple PulseTemplates that are defined on different channels into an AtomicPulseTemplate."""
    def __init__(self,
                 *subtemplates: Union[AtomicPulseTemplate, MappingTuple, MappingPulseTemplate],
                 external_parameters: Optional[Set[str]]=None,
                 identifier: Optional[str]=None,
                 parameter_constraints: Optional[List]=None,
                 measurements: Optional[List[MeasurementDeclaration]]=None,
                 registry: PulseRegistryType=None,
                 duration: Union[str, Expression, bool]=False) -> None:
        """Parallels multiple AtomicPulseTemplates of the same duration. The duration equality check is performed on
        construction by default. If the duration keyword argument is given the check is performed on instantiation
        (when build_waveform is called). duration can be a Expression to enforce a certain duration or True for an
        unspecified duration.

        Args:
            *subtemplates: Positional arguments are subtemplates to combine.
            identifier: Forwarded to AtomicPulseTemplate.__init__
            parameter_constraints: Forwarded to ParameterConstrainer.__init__
            measurements: Forwarded to AtomicPulseTemplate.__init__
            duration: Enforced duration of the pulse template on instantiation. build_waveform checks all sub-waveforms
            have this duration. If True the equality of durations is only checked durtin instantiation not construction.
            external_parameters: No functionality. (Deprecated)
        """
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

        if not duration:
            duration = self._subtemplates[0].duration
            for subtemplate in self._subtemplates[1:]:
                if almost_equal(duration.sympified_expression, subtemplate.duration.sympified_expression):
                    continue
                else:
                    raise ValueError('Could not assert duration equality of {} and {}'.format(duration,
                                                                                              subtemplate.duration))
            self._duration = None
        elif duration is True:
            self._duration = None
        else:
            self._duration = ExpressionScalar(duration)

        self._register(registry=registry)

    @property
    def duration(self) -> ExpressionScalar:
        if self._duration:
            return self._duration
        else:
            return self._subtemplates[0].duration

    @property
    def parameter_names(self) -> Set[str]:
        return set.union(self.measurement_parameters,
                         self.constrained_parameters,
                         *(st.parameter_names for st in self._subtemplates),
                         self._duration.variables if self._duration else ())

    @property
    def subtemplates(self) -> Sequence[Union[AtomicPulseTemplate, MappingPulseTemplate]]:
        return self._subtemplates

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return set.union(*(st.defined_channels for st in self._subtemplates))

    @property
    def measurement_names(self) -> Set[str]:
        return super().measurement_names.union(*(st.measurement_names for st in self._subtemplates))

    def build_waveform(self, parameters: Dict[str, numbers.Real],
                       channel_mapping: Dict[ChannelID, Optional[ChannelID]]) -> Optional[Waveform]:
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
            waveform = sub_waveforms[0]
        else:
            waveform = MultiChannelWaveform(sub_waveforms)

        if self._duration:
            expected_duration = self._duration.evaluate_numeric(**parameters)

            if not isclose(expected_duration, waveform.duration):
                raise ValueError('The duration does not '
                                 'equal the expected duration',
                                 expected_duration, waveform.duration)

        return waveform

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

    def get_serialization_data(self, serializer: Optional[Serializer]=None) -> Dict[str, Any]:
        data = super().get_serialization_data(serializer)
        data['subtemplates'] = self.subtemplates

        if serializer: # compatibility to old serialization routines, deprecated
            data = dict()
            data['subtemplates'] = [serializer.dictify(subtemplate) for subtemplate in self.subtemplates]

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