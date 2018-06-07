"""This module defines RepetitionPulseTemplate, a higher-order hierarchical pulse template that
represents the n-times repetition of another PulseTemplate."""

from typing import Dict, List, Set, Optional, Union, Any, Tuple, cast
from numbers import Real
from warnings import warn

import numpy as np

from qctoolkit.serialization import Serializer

from qctoolkit.utils.types import ChannelID, TimeType
from qctoolkit.expressions import ExpressionScalar
from qctoolkit.utils import checked_int_cast
from qctoolkit.pulses.pulse_template import PulseTemplate
from qctoolkit.pulses.loop_pulse_template import LoopPulseTemplate
from qctoolkit.pulses.sequencing import Sequencer
from qctoolkit.pulses.instructions import InstructionBlock, InstructionPointer, Waveform
from qctoolkit.pulses.parameters import Parameter, ParameterConstrainer, ParameterNotProvidedException
from qctoolkit.pulses.conditions import Condition
from qctoolkit.pulses.measurement import MeasurementDefiner, MeasurementDeclaration


__all__ = ["RepetitionPulseTemplate", "ParameterNotIntegerException"]


class RepetitionWaveform(Waveform):
    """This class allows putting multiple PulseTemplate together in one waveform on the hardware."""
    def __init__(self, body: Waveform, repetition_count: int):
        self._body = body
        self._repetition_count = checked_int_cast(repetition_count)
        if repetition_count < 1 or not isinstance(repetition_count, int):
            raise ValueError('Repetition count must be an integer >0')

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return self._body.defined_channels

    def unsafe_sample(self,
                      channel: ChannelID,
                      sample_times: np.ndarray,
                      output_array: Union[np.ndarray, None]=None) -> np.ndarray:
        if output_array is None:
            output_array = np.empty_like(sample_times)
        body_duration = self._body.duration
        time = 0
        for _ in range(self._repetition_count):
            end = time + body_duration
            indices = slice(*np.searchsorted(sample_times, (float(time), float(end)), 'left'))
            self._body.unsafe_sample(channel=channel,
                                     sample_times=sample_times[indices] - time,
                                     output_array=output_array[indices])
            time = end
        return output_array

    @property
    def compare_key(self) -> Tuple[Any, int]:
        return self._body.compare_key, self._repetition_count

    @property
    def duration(self) -> TimeType:
        return self._body.duration*self._repetition_count

    def unsafe_get_subset_for_channels(self, channels: Set[ChannelID]) -> 'RepetitionWaveform':
        return RepetitionWaveform(body=self._body.unsafe_get_subset_for_channels(channels),
                                  repetition_count=self._repetition_count)


class RepetitionPulseTemplate(LoopPulseTemplate, ParameterConstrainer, MeasurementDefiner):
    """Repeats a PulseTemplate a constant number of times (possibly determined by a parameter value).

    RepetitionPulseTemplate simply repeats the given body PulseTemplate with the same parameter set for the
    specified number of times. It does not provide a loop index to the subtemplate. If you need to loop over an integer
    range and provide an index to the repeated template (at the cost of sequencing performance), use
    :class:`~qctoolkit.pulses.loop_pulse_template.ForLoopPulseTemplate`.
    """

    def __init__(self,
                 body: PulseTemplate,
                 repetition_count: Union[int, str, ExpressionScalar],
                 identifier: Optional[str]=None,
                 *args,
                 parameter_constraints: Optional[List]=None,
                 measurements: Optional[List[MeasurementDeclaration]]=None
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

        if (repetition_count < 0) is True:
            raise ValueError('Repetition count may not be negative')

        self._repetition_count = repetition_count

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
        return self.body.parameter_names | set(self.repetition_count.variables) | self.constrained_parameters

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

        body_block = InstructionBlock()
        body_block.return_ip = InstructionPointer(instruction_block, len(instruction_block))

        try:
            real_parameters = {v: parameters[v].get_value() for v in self._repetition_count.variables}
        except KeyError:
            raise ParameterNotProvidedException(next(v for v in self.repetition_count.variables if v not in parameters))
        self.insert_measurement_instruction(instruction_block,
                                            parameters=parameters,
                                            measurement_mapping=measurement_mapping)
        instruction_block.add_instruction_repj(self.get_repetition_count_value(real_parameters), body_block)
        sequencer.push(self.body, parameters=parameters, conditions=conditions,
                       window_mapping=measurement_mapping, channel_mapping=channel_mapping, target_block=body_block)

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, Condition]) -> bool:
        return any(parameters[v].requires_stop for v in self.repetition_count.variables)

    def get_serialization_data(self, serializer: Optional[Serializer]=None) -> Dict[str, Any]:
        data = dict(
            body=self.body,
            repetition_count=self.repetition_count.original_expression
        )
        if self.parameter_constraints:
            data['parameter_constraints'] = [str(c) for c in self.parameter_constraints]
        if self.measurement_declarations:
            data['measurements'] = self.measurement_declarations

        if serializer: # compatibility to old serialization routines, deprecated
            data['body'] = serializer.dictify(self.body)

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
