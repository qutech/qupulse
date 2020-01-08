"""This module defines RepetitionPulseTemplate, a higher-order hierarchical pulse template that
represents the n-times repetition of another PulseTemplate."""

from typing import Dict, List, Set, Optional, Union, Any, Tuple, cast
from numbers import Real
from warnings import warn

import numpy as np

from qupulse.serialization import Serializer, PulseRegistryType
from qupulse._program._loop import Loop

from qupulse.utils.types import ChannelID
from qupulse.expressions import ExpressionScalar
from qupulse.utils import checked_int_cast
from qupulse.pulses.pulse_template import PulseTemplate
from qupulse.pulses.loop_pulse_template import LoopPulseTemplate
from qupulse.pulses.sequencing import Sequencer
from qupulse._program.instructions import InstructionBlock, InstructionPointer
from qupulse._program.waveforms import RepetitionWaveform
from qupulse.pulses.parameters import Parameter, ParameterConstrainer, ParameterNotProvidedException, MappedParameter
from qupulse.pulses.conditions import Condition
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
                 registry: PulseRegistryType=None
                 ) -> None:
        """Create a new RepetitionPulseTemplate instance.

        Args:
            body (PulseTemplate): The PulseTemplate which will be repeated.
            repetition_count (int or ParameterDeclaration): The number of repetitions either as a
                constant integer value or as a parameter declaration.
            identifier (str): A unique identifier for use in serialization. (optional)
        """
        if len(args) == 1 and parameter_constraints is None:
            warn('You used parameter_constraints as a positional argument. It will be keyword only in a future version.', DeprecationWarning)
        elif args:
            TypeError('RepetitionPulseTemplate expects 3 positional arguments, got ' + str(3 + len(args)))

        LoopPulseTemplate.__init__(self, identifier=identifier, body=body)
        ParameterConstrainer.__init__(self, parameter_constraints=parameter_constraints)
        MeasurementDefiner.__init__(self, measurements=measurements)

        repetition_count = ExpressionScalar.make(repetition_count)

        if repetition_count < 0:
            raise ValueError('Repetition count may not be negative')

        if repetition_count == 0:
            warn("Repetition pulse template with 0 repetitions on construction.")

        self._repetition_count = repetition_count

        self._register(registry=registry)

    @property
    def repetition_count(self) -> ExpressionScalar:
        """The amount of repetitions. Either a constant integer or a ParameterDeclaration object."""
        return self._repetition_count

    def get_repetition_count_value(self, parameters: Dict[str, Real]) -> int:
        value = self._repetition_count.evaluate_numeric(**parameters)
        try:
            return checked_int_cast(value)
        except ValueError:
            raise ParameterNotIntegerException(str(self._repetition_count), value)

    def __str__(self) -> str:
        return "RepetitionPulseTemplate: <{}> times <{}>"\
            .format(self._repetition_count, self.body)

    @property
    def parameter_names(self) -> Set[str]:
        return set.union(self.body.parameter_names, self.repetition_count.variables, self.constrained_parameters,
                         self.measurement_parameters)

    @property
    def measurement_names(self) -> Set[str]:
        return self.body.measurement_names | MeasurementDefiner.measurement_names.fget(self)

    @property
    def duration(self) -> ExpressionScalar:
        return self.repetition_count * self.body.duration

    def build_sequence(self,
                       sequencer: Sequencer,
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, Condition],
                       measurement_mapping: Dict[str, Optional[str]],
                       channel_mapping: Dict[ChannelID, Optional[ChannelID]],
                       instruction_block: InstructionBlock) -> None:
        self.validate_parameter_constraints(parameters=parameters)
        try:
            real_parameters = {v: parameters[v].get_value() for v in self._repetition_count.variables}
        except KeyError:
            raise ParameterNotProvidedException(next(v for v in self.repetition_count.variables if v not in parameters))

        self.insert_measurement_instruction(instruction_block,
                                            parameters=parameters,
                                            measurement_mapping=measurement_mapping)

        repetition_count = self.get_repetition_count_value(real_parameters)
        if repetition_count > 0:
            body_block = InstructionBlock()
            body_block.return_ip = InstructionPointer(instruction_block, len(instruction_block))

            instruction_block.add_instruction_repj(repetition_count, body_block)
            sequencer.push(self.body, parameters=parameters, conditions=conditions,
                           window_mapping=measurement_mapping, channel_mapping=channel_mapping, target_block=body_block)

    def _internal_create_program(self, *,
                                 parameters: Dict[str, Parameter],
                                 measurement_mapping: Dict[str, Optional[str]],
                                 channel_mapping: Dict[ChannelID, Optional[ChannelID]],
                                 global_transformation: Optional['Transformation'],
                                 to_single_waveform: Set[Union[str, 'PulseTemplate']],
                                 parent_loop: Loop,
                                 volatile: Set[str]) -> None:
        self.validate_parameter_constraints(parameters=parameters)
        relevant_params = set(self._repetition_count.variables).union(self.measurement_parameters)
        try:
            real_parameters = {v: parameters[v].get_value() for v in relevant_params}
        except KeyError as e:
            raise ParameterNotProvidedException(str(e)) from e

        repetition_count = max(0, self.get_repetition_count_value(real_parameters))

        # todo (2018-07-19): could in some circumstances possibly just multiply subprogram repetition count?
        # could be tricky if any repetition count is volatile ? check later and optimize if necessary
        if repetition_count > 0:
            if volatile.intersection(self.repetition_count.variables):
                repetition_expression = self.repetition_count.evaluate_symbolic(
                    {parameter: value for parameter, value in real_parameters.items()
                     if parameter not in volatile})
                repetition_parameter = MappedParameter(repetition_expression, parameters)
            else:
                repetition_parameter = None

            repj_loop = Loop(repetition_count=repetition_count, repetition_parameter=repetition_parameter)
            self.body._create_program(parameters=parameters,
                                      measurement_mapping=measurement_mapping,
                                      channel_mapping=channel_mapping,
                                      global_transformation=global_transformation,
                                      to_single_waveform=to_single_waveform,
                                      parent_loop=repj_loop,
                                      volatile=volatile)
            if repj_loop.waveform is not None or len(repj_loop.children) > 0:
                measurements = self.get_measurement_windows(real_parameters, measurement_mapping)
                if measurements:
                    parent_loop.add_measurements(measurements)

                parent_loop.append_child(loop=repj_loop)

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, Condition]) -> bool:
        return any(parameters[v].requires_stop for v in self.repetition_count.variables)

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
        return [self.repetition_count * c for c in body_integral]


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
