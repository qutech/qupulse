"""This module defines MultiChannelPulseTemplate, which allows the combination of several
AtomicPulseTemplates into a single template spanning several channels.

Classes:
    - MultiChannelPulseTemplate: A pulse template defined for several channels by combining pulse
        templates
    - MultiChannelWaveform: A waveform defined for several channels by combining waveforms
"""

from typing import Dict, List, Optional, Any, Iterable, Union, Set, Sequence, Mapping
import numbers
import warnings

import sympy

from qupulse.serialization import Serializer, PulseRegistryType
from qupulse.parameter_scope import Scope

from qupulse.utils import isclose
from qupulse.utils.sympy import almost_equal, Sympifyable
from qupulse.utils.types import ChannelID, FrozenDict, TimeType
from qupulse.utils.numeric import are_durations_compatible
from qupulse._program.waveforms import MultiChannelWaveform, Waveform, TransformingWaveform
from qupulse._program.transformation import ParallelConstantChannelTransformation, Transformation, chain_transformations
from qupulse.pulses.pulse_template import PulseTemplate, AtomicPulseTemplate
from qupulse.pulses.mapping_pulse_template import MappingPulseTemplate, MappingTuple
from qupulse.pulses.parameters import Parameter, ParameterConstrainer
from qupulse.pulses.measurement import MeasurementDeclaration, MeasurementWindow
from qupulse.expressions import Expression, ExpressionScalar, ExpressionLike

__all__ = ["AtomicMultiChannelPulseTemplate", "ParallelConstantChannelPulseTemplate"]


class AtomicMultiChannelPulseTemplate(AtomicPulseTemplate, ParameterConstrainer):
    def __init__(self,
                 *subtemplates: Union[AtomicPulseTemplate, MappingTuple, MappingPulseTemplate],
                 identifier: Optional[str]=None,
                 parameter_constraints: Optional[List]=None,
                 measurements: Optional[List[MeasurementDeclaration]]=None,
                 registry: PulseRegistryType=None,
                 duration: Optional[ExpressionLike] = None,
                 pad_values: Mapping[ChannelID, ExpressionLike] = None) -> None:
        """Parallels multiple AtomicPulseTemplates that are defined on different channels. The `duration` and
        `pad_values` arguments can be used to determine how differences in the sub-templates' durations are handled.

        `duration` is True:
            There are no compatibility checks performed during the initialization of this object.
        `duration` is None (default):
            The durations may not be incompatible if it can be determined



        equality check is performed on
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

        if duration in (False, True):
            warnings.warn('Boolean duration is deprecated since qupulse 0.6', DeprecationWarning)
            duration = None

        self._subtemplates = [st if isinstance(st, PulseTemplate) else MappingPulseTemplate.from_tuple(st) for st in
                              subtemplates]

        if duration is None:
            self._duration = None
        else:
            self._duration = ExpressionScalar(duration)

        if pad_values is None:
            self._pad_values = FrozenDict()
        else:
            self._pad_values = FrozenDict((ch, None if value is None else ExpressionScalar(value))
                                          for ch, value in pad_values.items())

        if not self._subtemplates:
            raise ValueError('Cannot create empty MultiChannelPulseTemplate')

        if self._pad_values.keys() - self.defined_channels:
            raise ValueError('Padding value for channels not defined in subtemplates',
                             self._pad_values.keys() - self.defined_channels)

        # factored out for easier readability
        # important that asserts happen before register
        self._assert_atomic_sub_templates()
        self._assert_disjoint_channels()
        self._assert_compatible_durations()

        self._register(registry=registry)

    def _assert_atomic_sub_templates(self):
        for sub_template in self._subtemplates:
            template = sub_template
            while isinstance(template, MappingPulseTemplate):
                template = template.template

            if not isinstance(template, AtomicPulseTemplate):
                if template is sub_template:
                    raise TypeError('Non atomic subtemplate: {}'.format(template))
                else:
                    raise TypeError('Non atomic subtemplate of MappingPulseTemplate: {}'.format(template))

    def _assert_disjoint_channels(self):
        defined_channels = [st.defined_channels for st in self._subtemplates]

        # check there are no intersections between channels
        for i, channels_i in enumerate(defined_channels):
            for j, channels_j in enumerate(defined_channels[i + 1:]):
                if channels_i & channels_j:
                    raise ChannelMappingException('subtemplate {}'.format(i + 1),
                                                  'subtemplate {}'.format(i + 2 + j),
                                                  (channels_i & channels_j).pop())

    def _assert_compatible_durations(self):
        """Check if we can prove that durations of unpadded waveforms are incompatible."""
        unpadded_durations = [sub_template.duration
                              for sub_template in self._subtemplates
                              if sub_template.defined_channels - self._pad_values.keys()]
        are_compatible = are_durations_compatible(self.duration, *unpadded_durations)
        if are_compatible is False:
            # durations definitely not compatible
            raise ValueError('Durations are definitely not compatible: {}'.format(unpadded_durations),
                             unpadded_durations)

    @property
    def duration(self) -> ExpressionScalar:
        if self._duration:
            return self._duration
        else:
            return ExpressionScalar(sympy.Max(*(subtemplate.duration for subtemplate in self._subtemplates)))

    @property
    def parameter_names(self) -> Set[str]:
        return set.union(self.measurement_parameters,
                         self.constrained_parameters,
                         *(st.parameter_names for st in self._subtemplates),
                         self._duration.variables if self._duration else (),
                         *(value.variables for value in self._pad_values.values() if value is not None))

    @property
    def subtemplates(self) -> Sequence[Union[AtomicPulseTemplate, MappingPulseTemplate]]:
        return self._subtemplates

    @property
    def pad_values(self) -> Mapping[ChannelID, Optional[Expression]]:
        return self._pad_values

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return set.union(*(st.defined_channels for st in self._subtemplates))

    @property
    def measurement_names(self) -> Set[str]:
        return super().measurement_names.union(*(st.measurement_names for st in self._subtemplates))

    def build_waveform(self, parameters: Dict[str, numbers.Real],
                       channel_mapping: Dict[ChannelID, Optional[ChannelID]]) -> Optional[Waveform]:
        self.validate_parameter_constraints(parameters=parameters, volatile=set())

        sub_waveforms = []
        for subtemplate in self.subtemplates:
            sub_waveform = subtemplate.build_waveform(parameters,
                                                      channel_mapping=channel_mapping)
            if sub_waveform is not None:
                sub_waveforms.append(sub_waveform)

        pad_values = {}
        for ch, pad_expression in self._pad_values.items():
            ch = channel_mapping[ch]
            if ch is None:
                continue
            elif pad_expression is None:
                pad_values[ch] = None
            else:
                pad_values[ch] = pad_expression.evaluate_in_scope(parameters)

        if len(sub_waveforms) == 0:
            return None

        if self._duration is None:
            duration = None
        else:
            duration = TimeType.from_float(self._duration.evaluate_numeric(**parameters))

        if len(sub_waveforms) == 1 and (duration in (None, sub_waveforms[0].duration)):
            # No padding
            waveform = sub_waveforms[0]
        else:
            waveform = MultiChannelWaveform.from_iterable(sub_waveforms, pad_values, duration=duration)

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
        if self._pad_values:
            data['pad_values'] = self._pad_values
        if self._duration is not None:
            data['duration'] = self._duration

        return data

    @classmethod
    def deserialize(cls, serializer: Optional[Serializer]=None, **kwargs) -> 'AtomicMultiChannelPulseTemplate':
        subtemplates = kwargs['subtemplates']
        del kwargs['subtemplates']

        if serializer:  # compatibility to old serialization routines, deprecated
            subtemplates = [serializer.deserialize(st) for st in subtemplates]

        return cls(*subtemplates, **kwargs)

    @property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        t = self._AS_EXPRESSION_TIME
        self_duration = self.duration.underlying_expression
        result = self._as_expression()
        for ch, expr in result.items():
            ch_integral = sympy.integrate(expr.underlying_expression, (t, 0, self_duration))
            result[ch] = ExpressionScalar(ch_integral)
        return result

    def _as_expression(self) -> Dict[ChannelID, ExpressionScalar]:
        t = self._AS_EXPRESSION_TIME
        as_expression = {}
        for sub_template in self.subtemplates:
            sub_duration = sub_template.duration.sympified_expression
            sub_as_expression = sub_template._as_expression()

            padding = t > sub_duration

            for ch, ch_expr in sub_as_expression.items():
                pad_value = self._pad_values.get(ch, None)
                if pad_value is None:
                    pad_value = ch_expr.underlying_expression.subs({t: sub_duration})
                as_expression[ch] = ExpressionScalar(sympy.Piecewise((pad_value, padding),
                                                                     (ch_expr.underlying_expression, True)))
        return as_expression


class ParallelConstantChannelPulseTemplate(PulseTemplate):
    def __init__(self,
                 template: PulseTemplate,
                 overwritten_channels: Mapping[ChannelID, Union[ExpressionScalar, Sympifyable]], *,
                 identifier: Optional[str]=None,
                 registry: Optional[PulseRegistryType]=None):
        super().__init__(identifier=identifier)

        self._template = template
        self._overwritten_channels = {channel: ExpressionScalar(value)
                                      for channel, value in overwritten_channels.items()}

        self._register(registry=registry)

    @property
    def template(self) -> PulseTemplate:
        return self._template

    @property
    def overwritten_channels(self) -> Mapping[str, ExpressionScalar]:
        return self._overwritten_channels

    def _get_overwritten_channels_values(self,
                                         parameters: Mapping[str, Union[numbers.Real]]
                                         ) -> Dict[str, numbers.Real]:
        return {name: value.evaluate_in_scope(parameters)
                for name, value in self.overwritten_channels.items()}

    def _internal_create_program(self, *,
                                 scope: Scope,
                                 global_transformation: Optional[Transformation],
                                 **kwargs):
        overwritten_channels = self._get_overwritten_channels_values(parameters=scope)
        transformation = ParallelConstantChannelTransformation(overwritten_channels)

        if global_transformation is not None:
            transformation = chain_transformations(global_transformation, transformation)

        self._template._create_program(scope=scope,
                                       global_transformation=transformation,
                                       **kwargs)

    def build_waveform(self, parameters: Dict[str, numbers.Real],
                       channel_mapping: Dict[ChannelID, Optional[ChannelID]]) -> Optional[Waveform]:
        inner_waveform = self._template.build_waveform(parameters, channel_mapping)

        if inner_waveform:
            overwritten_channels = self._get_overwritten_channels_values(parameters=parameters)
            transformation = ParallelConstantChannelTransformation(overwritten_channels)
            return TransformingWaveform(inner_waveform, transformation)

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return set.union(self._template.defined_channels, self._overwritten_channels.keys())

    @property
    def measurement_names(self) -> Set[str]:
        return self._template.measurement_names

    @property
    def transformation_parameters(self) -> Set[str]:
        return set.union(*(set(value.variables) for value in self.overwritten_channels.values()))

    @property
    def parameter_names(self):
        return self._template.parameter_names | self.transformation_parameters

    @property
    def duration(self) -> ExpressionScalar:
        return self.template.duration

    @property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        integral = self._template.integral

        duration = self._template.duration
        for channel, value in self._overwritten_channels.items():
            integral[channel] = value * duration
        return integral

    def get_serialization_data(self, serializer: Optional[Serializer]=None) -> Dict[str, Any]:
        if serializer:
            raise NotImplementedError('Legacy serialization not implemented for new class')

        data = super().get_serialization_data()
        data['template'] = self._template
        data['overwritten_channels'] = self._overwritten_channels
        return data


class ChannelMappingException(Exception):
    def __init__(self, obj1, obj2, intersect_set):
        self.intersect_set = intersect_set
        self.obj1 = obj1
        self.obj2 = obj2

    def __str__(self) -> str:
        return 'Channel <{chs}> is defined in {o1} and {o2}'.format(chs=self.intersect_set, o1=self.obj1, o2=self.obj2)
