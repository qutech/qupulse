# SPDX-FileCopyrightText: 2014-2024 Quantum Technology Group and Chair of Software Engineering, RWTH Aachen University
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""This module defines MultiChannelPulseTemplate, which allows the combination of several
AtomicPulseTemplates into a single template spanning several channels.

Classes:
    - MultiChannelPulseTemplate: A pulse template defined for several channels by combining pulse
        templates
    - ParallelChannelPulseTemplate: A pulse template to add channels to an existing pulse template.
"""

from typing import Dict, List, Optional, Any, AbstractSet, Union, Set, Sequence, Mapping
import numbers
import warnings

from qupulse.serialization import Serializer, PulseRegistryType
from qupulse.parameter_scope import Scope

from qupulse.utils import isclose
from qupulse.utils.sympy import almost_equal, Sympifyable
from qupulse.utils.types import ChannelID, TimeType
from qupulse.program.waveforms import MultiChannelWaveform, Waveform, TransformingWaveform
from qupulse.program.transformation import ParallelChannelTransformation, Transformation, chain_transformations
from qupulse.pulses.pulse_template import PulseTemplate, AtomicPulseTemplate
from qupulse.pulses.mapping_pulse_template import MappingPulseTemplate, MappingTuple
from qupulse.pulses.parameters import ParameterConstrainer
from qupulse.pulses.measurement import MeasurementDeclaration, MeasurementWindow
from qupulse.expressions import Expression, ExpressionScalar, ExpressionLike

__all__ = ["AtomicMultiChannelPulseTemplate", "ParallelConstantChannelPulseTemplate"]


class AtomicMultiChannelPulseTemplate(AtomicPulseTemplate, ParameterConstrainer):
    def __init__(self,
                 *subtemplates: Union[AtomicPulseTemplate, MappingTuple, MappingPulseTemplate],
                 identifier: Optional[str] = None,
                 parameter_constraints: Optional[List] = None,
                 measurements: Optional[List[MeasurementDeclaration]] = None,
                 registry: PulseRegistryType = None,
                 duration: Optional[ExpressionLike] = None) -> None:
        """Combines multiple AtomicPulseTemplates of the same duration that are defined on different channels into an
        AtomicPulseTemplate.
        If the duration keyword argument is given it is enforced that the instantiated pulse template has this duration.
        If duration is None the duration of the PT is the duration of the first subtemplate.
        There are probably changes to this behaviour in the future.

        Args:
            *subtemplates: Positional arguments are subtemplates to combine.
            identifier: Forwarded to AtomicPulseTemplate.__init__
            parameter_constraints: Forwarded to ParameterConstrainer.__init__
            measurements: Forwarded to AtomicPulseTemplate.__init__
            duration: Enforced duration of the pulse template on instantiation. build_waveform checks all sub-waveforms
            have this duration. If True the equality of durations is only checked durtin instantiation not construction.
        """
        AtomicPulseTemplate.__init__(self, identifier=identifier, measurements=measurements)
        ParameterConstrainer.__init__(self, parameter_constraints=parameter_constraints)

        self._subtemplates = [st if isinstance(st, PulseTemplate) else MappingPulseTemplate.from_tuple(st) for st in
                              subtemplates]

        if duration in (True, False):
            warnings.warn("Boolean duration is deprecated since qupulse 0.6 and interpreted as None",
                          category=DeprecationWarning, stacklevel=2)
            duration = None

        for subtemplate in self._subtemplates:
            if not subtemplate._is_atomic():
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
                                                  (channels_i & channels_j).pop())

        if duration is None:
            self._duration = None
        else:
            self._duration = ExpressionScalar(duration)

        self._register(registry=registry)

    def with_parallel_atomic(self, *parallel: 'AtomicPulseTemplate') -> 'AtomicPulseTemplate':
        from qupulse.pulses import AtomicMultiChannelPT
        if parallel:
            if self.identifier:
                return AtomicMultiChannelPT(self, *parallel)
            else:
                return AtomicMultiChannelPT(
                    *self._subtemplates, *parallel,
                    measurements=self.measurement_declarations,
                    parameter_constraints=self.parameter_constraints,
                )
        else:
            return self

    @property
    def duration(self) -> ExpressionScalar:
        if self._duration is None:
            return self._subtemplates[0].duration
        else:
            return self._duration

    @property
    def parameter_names(self) -> AbstractSet[str]:
        return set().union(self.measurement_parameters,
                           self.constrained_parameters,
                           *(st.parameter_names for st in self._subtemplates),
                           getattr(self._duration, 'variables', ()))

    @property
    def subtemplates(self) -> Sequence[Union[AtomicPulseTemplate, MappingPulseTemplate]]:
        return self._subtemplates

    @property
    def defined_channels(self) -> AbstractSet[ChannelID]:
        return set().union(*(st.defined_channels for st in self._subtemplates))

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

        if len(sub_waveforms) == 0:
            return None

        if len(sub_waveforms) == 1:
            waveform = sub_waveforms[0]
        else:
            waveform = MultiChannelWaveform.from_parallel(sub_waveforms)

        if self._duration:
            expected_duration = self._duration.evaluate_in_scope(parameters)

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

        if serializer:  # compatibility to old serialization routines, deprecated
            subtemplates = [serializer.deserialize(st) for st in subtemplates]

        return cls(*subtemplates, **kwargs)

    @property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        expressions = dict()
        for subtemplate in self._subtemplates:
            expressions.update(subtemplate.integral)
        return expressions

    def _as_expression(self) -> Dict[ChannelID, ExpressionScalar]:
        expressions = dict()
        for subtemplate in self._subtemplates:
            expressions.update(subtemplate._as_expression())
        return expressions

    @property
    def initial_values(self) -> Dict[ChannelID, ExpressionScalar]:
        values = {}
        for subtemplate in self._subtemplates:
            values.update(subtemplate.initial_values)
        return values

    @property
    def final_values(self) -> Dict[ChannelID, ExpressionScalar]:
        values = {}
        for subtemplate in self._subtemplates:
            values.update(subtemplate.final_values)
        return values


class ParallelChannelPulseTemplate(PulseTemplate):
    def __init__(self,
                 template: PulseTemplate,
                 overwritten_channels: Mapping[ChannelID, Union[ExpressionScalar, Sympifyable]], *,
                 identifier: Optional[str]=None,
                 registry: Optional[PulseRegistryType] = None):
        """Pulse template to add new or overwrite existing channels of a contained pulse template. The channel values
        may be time dependent if the contained pulse template is atomic.

        Args:
            template: Inner pulse template where all channels that are not overwritten will stay the same.
            overwritten_channels: Mapping of channels to values that this channel will have. This can overwrite existing
            channels or add new ones. May be time dependent if template is atomic.
            identifier: Name of the pulse template for serialization
            registry: Pulse template gets registered here if not None.
        """
        super().__init__(identifier=identifier)

        self._template = template
        self._overwritten_channels = {channel: ExpressionScalar(value)
                                      for channel, value in overwritten_channels.items()}

        if not template._is_atomic():
            for expr in self._overwritten_channels.values():
                if 't' in expr.variables:
                    raise TypeError(f"{type(self).__name__} currently only supports time dependent expressions if the "
                                    f"pulse template is atomic.", self)

        self._register(registry=registry)

    @property
    def template(self) -> PulseTemplate:
        return self._template

    @property
    def overwritten_channels(self) -> Mapping[str, ExpressionScalar]:
        return self._overwritten_channels

    def _get_overwritten_channels_values(self,
                                         parameters: Mapping[str, Union[numbers.Real]],
                                         channel_mapping: Dict[ChannelID, Optional[ChannelID]]
                                         ) -> Dict[str, Union[numbers.Real, ExpressionScalar]]:
        """Return a dictionary of ChannelID to channel value mappings. The channel values can bei either numbers or time
        dependent expressions."""
        return {channel_mapping[name]: value.evaluate_symbolic(parameters) if 't' in value.variables else value.evaluate_in_scope(parameters)
                for name, value in self.overwritten_channels.items()
                if channel_mapping[name] is not None}

    def _internal_create_program(self, *,
                                 scope: Scope,
                                 global_transformation: Optional[Transformation],
                                 channel_mapping: Dict[ChannelID, Optional[ChannelID]],
                                 **kwargs):
        overwritten_channels = self._get_overwritten_channels_values(parameters=scope, channel_mapping=channel_mapping)
        transformation = ParallelChannelTransformation(overwritten_channels)

        if global_transformation is not None:
            transformation = chain_transformations(global_transformation, transformation)

        self._template._create_program(scope=scope,
                                       channel_mapping=channel_mapping,
                                       global_transformation=transformation,
                                       **kwargs)

    def build_waveform(self, parameters: Dict[str, numbers.Real],
                       channel_mapping: Dict[ChannelID, Optional[ChannelID]]) -> Optional[Waveform]:
        inner_waveform = self._template.build_waveform(parameters, channel_mapping)

        if inner_waveform:
            overwritten_channels = self._get_overwritten_channels_values(parameters=parameters,
                                                                         channel_mapping=channel_mapping)
            transformation = ParallelChannelTransformation(overwritten_channels)
            return TransformingWaveform.from_transformation(inner_waveform, transformation)

    @property
    def defined_channels(self) -> AbstractSet[ChannelID]:
        return set().union(self._template.defined_channels, self._overwritten_channels.keys())

    @property
    def measurement_names(self) -> AbstractSet[str]:
        return self._template.measurement_names

    @property
    def transformation_parameters(self) -> AbstractSet[str]:
        return set().union(*(value.variables for value in self.overwritten_channels.values())) - {'t'}

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

    @property
    def initial_values(self) -> Dict[ChannelID, ExpressionScalar]:
        values = self._template.initial_values
        values.update(self._overwritten_channels)
        return values

    @property
    def final_values(self) -> Dict[ChannelID, ExpressionScalar]:
        values = self._template.final_values
        values.update(self._overwritten_channels)
        return values

    def get_serialization_data(self, serializer: Optional[Serializer]=None) -> Dict[str, Any]:
        if serializer:
            raise NotImplementedError('Legacy serialization not implemented for new class')

        data = super().get_serialization_data()
        data['template'] = self._template
        data['overwritten_channels'] = self._overwritten_channels
        return data

    def with_parallel_channels(self, values: Mapping[ChannelID, ExpressionLike]) -> 'PulseTemplate':
        if self.identifier:
            return super().with_parallel_channels(values)
        else:
            return ParallelConstantChannelPulseTemplate(
                self._template,
                {**self._overwritten_channels, **values},
            )

    def _is_atomic(self) -> bool:
        return self._template._is_atomic()


ParallelConstantChannelPulseTemplate = ParallelChannelPulseTemplate


class ChannelMappingException(Exception):
    def __init__(self, obj1, obj2, intersect_set):
        self.intersect_set = intersect_set
        self.obj1 = obj1
        self.obj2 = obj2

    def __str__(self) -> str:
        return 'Channel <{chs}> is defined in {o1} and {o2}'.format(chs=self.intersect_set, o1=self.obj1, o2=self.obj2)
