# SPDX-FileCopyrightText: 2014-2024 Quantum Technology Group and Chair of Software Engineering, RWTH Aachen University
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""This module defines RepetitionPulseTemplate, a higher-order hierarchical pulse template that
represents the n-times repetition of another PulseTemplate."""

from typing import Dict, List, AbstractSet, Optional, Union, Any, Mapping, cast
from numbers import Real
from warnings import warn

import numpy as np

from qupulse.serialization import Serializer, PulseRegistryType
from qupulse.program.volatile import VolatileRepetitionCount
from qupulse.program import ProgramBuilder
from qupulse.parameter_scope import Scope

from qupulse.utils.types import ChannelID
from qupulse.expressions import ExpressionScalar
from qupulse.utils import checked_int_cast
from qupulse.pulses.pulse_template import PulseTemplate
from qupulse.pulses.metadata import TemplateMetadata
from qupulse.pulses.loop_pulse_template import LoopPulseTemplate
from qupulse.pulses.parameters import ParameterConstrainer
from qupulse.pulses.measurement import MeasurementDefiner, MeasurementDeclaration


__all__ = ["RepetitionPulseTemplate", "ParameterNotIntegerException"]


class RepetitionPulseTemplate(LoopPulseTemplate, ParameterConstrainer, MeasurementDefiner):
    """Repeats a PulseTemplate a constant number of times (possibly determined by a parameter value).

    RepetitionPulseTemplate simply repeats the given body PulseTemplate with the same parameter set for the
    specified number of times. It does not provide a loop index to the subtemplate. If you need to loop over an integer
    range and provide an index to the repeated template (at the cost of sequencing performance), use
    :class:`~qupulse.pulses.loop_pulse_template.ForLoopPulseTemplate`.
    """

    def __init__(self,
                 body: PulseTemplate,
                 repetition_count: Union[int, str, ExpressionScalar],
                 identifier: Optional[str]=None,
                 *args,
                 parameter_constraints: Optional[List]=None,
                 measurements: Optional[List[MeasurementDeclaration]]=None,
                 metadata: Union[TemplateMetadata, dict] = None,
                 registry: PulseRegistryType=None
                 ) -> None:
        """
        Furthermore, this class allows associating an identifier, measurements, and parameter constraints with this sequence.
        If the body evaluates to nothing during instantiation, the associated measurements are dropped.

        Translation into a single waveform can be forced by passing ``to_single_waveform == 'always'`` in the ``metadata``.

        The default creation does not flatten multiple nested repetition pulse templates.
        Use :py:meth:`.RepetitionPulseTemplate.with_repetition` which will do that if ``identifier`` and ```metadata`` are not set.

        Raises:
            ValueError: If the repetition count is negative

        Args:
            body: The PulseTemplate which will be repeated.
            repetition_count: The number of repetitions.
            identifier: A unique identifier for use in serialization.
            parameter_constraints: See :py:class:`.ParameterConstrainer` for details
            metadata: Used to initialize :py:attr:`.PulseTemplate.metadata`
            registry: This pulse template registers itself there under the given identifier if supplied.
        """
        if len(args) == 1 and parameter_constraints is None:
            warn('You used parameter_constraints as a positional argument. It will be keyword only in a future version.', DeprecationWarning)
        elif args:
            TypeError('RepetitionPulseTemplate expects 3 positional arguments, got ' + str(3 + len(args)))

        LoopPulseTemplate.__init__(self, identifier=identifier, body=body, metadata=metadata)
        ParameterConstrainer.__init__(self, parameter_constraints=parameter_constraints)
        MeasurementDefiner.__init__(self, measurements=measurements)

        repetition_count = ExpressionScalar.make(repetition_count)

        if repetition_count < 0:
            raise ValueError('Repetition count may not be negative')

        if repetition_count == 0:
            warn("Repetition pulse template with 0 repetitions on construction.")

        self._repetition_count = repetition_count

        self._register(registry=registry)

    def with_repetition(self, repetition_count: Union[int, str, ExpressionScalar]) -> 'RepetitionPulseTemplate':
        repetition_count = ExpressionScalar.make(repetition_count)
        if self.identifier or self.metadata:
            return RepetitionPulseTemplate(self, repetition_count)
        else:
            return RepetitionPulseTemplate(
                self.body,
                self.repetition_count * repetition_count,
                parameter_constraints=self.parameter_constraints,
                measurements=self.measurement_declarations
            )

    @property
    def repetition_count(self) -> ExpressionScalar:
        """The amount of repetitions. Either a constant integer or a ParameterDeclaration object."""
        return self._repetition_count

    def get_repetition_count_value(self, parameters: Mapping[str, Real]) -> int:
        value = self._repetition_count.evaluate_in_scope(parameters)
        try:
            return checked_int_cast(value)
        except ValueError:
            raise ParameterNotIntegerException(str(self._repetition_count), value)

    def __str__(self) -> str:
        return "RepetitionPulseTemplate: <{}> times <{}>"\
            .format(self._repetition_count, self.body)

    @property
    def parameter_names(self) -> AbstractSet[str]:
        return set().union(self.body.parameter_names,
                           self.constrained_parameters,
                           self.measurement_parameters,
                           self.repetition_count.variables)

    @property
    def measurement_names(self) -> AbstractSet[str]:
        return self.body.measurement_names | MeasurementDefiner.measurement_names.fget(self)

    @property
    def duration(self) -> ExpressionScalar:
        return self.repetition_count * self.body.duration

    def _internal_build_program(self, program_builder: ProgramBuilder):
        build_context = program_builder.build_context
        scope = build_context.scope

        repetition_count = self.get_repetition_count_value(scope)
        if not (repetition_count > 0):
            return

        if scope.get_volatile_parameters().keys() & self.repetition_count.variables:
            repetition_definition = VolatileRepetitionCount(self.repetition_count, scope)
            assert int(repetition_definition) == repetition_count
        else:
            repetition_definition = repetition_count

        measurements = self.get_measurement_windows(scope, build_context.measurement_mapping)
        for repetition_program_builder in program_builder.with_repetition(repetition_definition, measurements=measurements):
            self.body._build_program(repetition_program_builder)



    def _internal_create_program(self, *,
                                 scope: Scope,
                                 measurement_mapping: Dict[str, Optional[str]],
                                 channel_mapping: Dict[ChannelID, Optional[ChannelID]],
                                 global_transformation: Optional['Transformation'],
                                 to_single_waveform: AbstractSet[Union[str, 'PulseTemplate']],
                                 program_builder: ProgramBuilder) -> None:
        self.validate_scope(scope)

        repetition_count = max(0, self.get_repetition_count_value(scope))

        if repetition_count > 0:
            if scope.get_volatile_parameters().keys() & self.repetition_count.variables:
                repetition_definition = VolatileRepetitionCount(self.repetition_count, scope)
                assert int(repetition_definition) == repetition_count
            else:
                repetition_definition = repetition_count

            measurements = self.get_measurement_windows(scope, measurement_mapping)

            for repetition_program_builder in program_builder.with_repetition(repetition_definition,
                                                                              measurements=measurements):
                self.body._create_program(scope=repetition_program_builder.build_context.scope,
                                          measurement_mapping=measurement_mapping,
                                          channel_mapping=channel_mapping,
                                          global_transformation=global_transformation,
                                          to_single_waveform=to_single_waveform,
                                          program_builder=repetition_program_builder)

    def get_serialization_data(self, serializer: Optional[Serializer]=None) -> Dict[str, Any]:
        data = super().get_serialization_data(serializer)
        data['body'] = self.body

        if serializer: # compatibility to old serialization routines, deprecated
            data = dict()
            data['body'] = serializer.dictify(self.body)

        data['repetition_count'] = self.repetition_count.original_expression

        if self.parameter_constraints:
            data['parameter_constraints'] = [str(c) for c in self.parameter_constraints]
        if self.measurement_declarations:
            data['measurements'] = self.measurement_declarations

        return data

    @classmethod
    def deserialize(cls, serializer: Optional[Serializer]=None, **kwargs) -> 'RepetitionPulseTemplate':
        if serializer: # compatibility to old serialization routines, deprecated
            kwargs['body'] = cast(PulseTemplate, serializer.deserialize(kwargs['body']))

        return super().deserialize(**kwargs)

    @property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        body_integral = self.body.integral
        return {channel: self.repetition_count * value for channel, value in body_integral.items()}

    @property
    def initial_values(self) -> Dict[ChannelID, ExpressionScalar]:
        return self.body.initial_values

    @property
    def final_values(self) -> Dict[ChannelID, ExpressionScalar]:
        return self.body.final_values


class ParameterNotIntegerException(Exception):
    """Indicates that the value of the parameter given as repetition count was not an integer."""

    def __init__(self, parameter_name: str, parameter_value: Any) -> None:
        super().__init__()
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value

    def __str__(self) -> str:
        return "The parameter <{}> must have an integral value (was <{}>) " \
            "for use as repetition count.".format(
                self.parameter_name,
                self.parameter_value
            )
