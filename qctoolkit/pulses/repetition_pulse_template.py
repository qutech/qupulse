"""This module defines RepetitionPulseTemplate, a higher-order hierarchical pulse template that
represents the n-times repetition of another PulseTemplate."""

from typing import Dict, List, Set, Optional, Union, Any, Iterable, Tuple
from numbers import Real

import numpy as np

from qctoolkit.serialization import Serializer

from qctoolkit.utils.types import MeasurementWindow, ChannelID
from qctoolkit.expressions import Expression
from qctoolkit.utils import checked_int_cast
from qctoolkit.pulses.pulse_template import PulseTemplate
from qctoolkit.pulses.loop_pulse_template import LoopPulseTemplate
from qctoolkit.pulses.sequencing import Sequencer
from qctoolkit.pulses.instructions import InstructionBlock, InstructionPointer, Waveform
from qctoolkit.pulses.parameters import Parameter, ParameterConstrainer, ParameterNotProvidedException
from qctoolkit.pulses.conditions import Condition


__all__ = ["RepetitionPulseTemplate", "ParameterNotIntegerException"]


class RepetitionWaveform(Waveform):
    """This class allows putting multiple PulseTemplate together in one waveform on the hardware."""
    def __init__(self, body: Waveform, repetition_count: int):
        self._body = body
        self._repetition_count = repetition_count
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
            output_array = np.empty(len(sample_times))
        body_duration = self._body.duration
        for i in range(self._repetition_count):
            indices = slice(*np.searchsorted(sample_times, (i*body_duration, (i+1)*body_duration)))
            self._body.unsafe_sample(channel=channel,
                                     sample_times=sample_times[indices],
                                     output_array=output_array[indices])
        return output_array

    @property
    def compare_key(self) -> Tuple[Any, int]:
        return self._body.compare_key, self._repetition_count

    @property
    def duration(self) -> float:
        return self._body.duration*self._repetition_count

    def get_measurement_windows(self) -> Iterable[MeasurementWindow]:
        def get_measurement_window_generator(body: Waveform, repetition_count: int):
            body_windows = list(body.get_measurement_windows())
            for i in range(repetition_count):
                for (name, begin, length) in body_windows:
                    yield (name, begin+i*body.duration, length)
        return get_measurement_window_generator(self._body, self._repetition_count)

    def unsafe_get_subset_for_channels(self, channels: Set[ChannelID]) -> 'RepetitionWaveform':
        return RepetitionWaveform(body=self._body.unsafe_get_subset_for_channels(channels),
                                  repetition_count=self._repetition_count)


class RepetitionPulseTemplate(LoopPulseTemplate, ParameterConstrainer):
    """Repeat a PulseTemplate a constant number of times.

    The equivalent to a simple for-loop in common programming languages in qctoolkit's pulse
    modelling.
    """

    def __init__(self,
                 body: PulseTemplate,
                 repetition_count: Union[int, str, Expression],
                 identifier: Optional[str]=None,
                 parameter_constraints: Optional[List]=None) -> None:
        """Create a new RepetitionPulseTemplate instance.

        Args:
            body (PulseTemplate): The PulseTemplate which will be repeated.
            repetition_count (int or ParameterDeclaration): The number of repetitions either as a
                constant integer value or as a parameter declaration.
            identifier (str): A unique identifier for use in serialization. (optional)
        """
        LoopPulseTemplate.__init__(self, identifier=identifier, body=body)
        ParameterConstrainer.__init__(self, parameter_constraints=parameter_constraints)

        if not isinstance(repetition_count, Expression):
            repetition_count = Expression(repetition_count)

        if (repetition_count < 0) is True:
            raise ValueError('Repetition count may not be negative')

        self._repetition_count = repetition_count

    @property
    def repetition_count(self) -> Expression:
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
        return self.body.parameter_names | set(self.repetition_count.variables)

    @property
    def measurement_names(self) -> Set[str]:
        return self.body.measurement_names

    @property
    def duration(self) -> Expression:
        return Expression(self.repetition_count.sympified_expression * self.body.duration.sympified_expression)

    def build_sequence(self,
                       sequencer: Sequencer,
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, Condition],
                       measurement_mapping: Dict[str, str],
                       channel_mapping: Dict[ChannelID, ChannelID],
                       instruction_block: InstructionBlock) -> None:
        self.validate_parameter_constraints(parameters=parameters)

        body_block = InstructionBlock()
        body_block.return_ip = InstructionPointer(instruction_block, len(instruction_block))

        try:
            real_parameters = {v: parameters[v].get_value() for v in self._repetition_count.variables}
        except KeyError:
            raise ParameterNotProvidedException(next(v for v in self.repetition_count.variables if v not in parameters))

        instruction_block.add_instruction_repj(self.get_repetition_count_value(real_parameters), body_block)
        sequencer.push(self.body, parameters=parameters, conditions=conditions,
                       window_mapping=measurement_mapping, channel_mapping=channel_mapping, target_block=body_block)

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, Condition]) -> bool:
        return any(parameters[v].requires_stop for v in self.repetition_count.variables)

    def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
        return dict(
            body=serializer.dictify(self.body),
            repetition_count=self.repetition_count.original_expression,
            parameter_constraints=self.parameter_constraints
        )

    @staticmethod
    def deserialize(serializer: Serializer,
                    repetition_count: Union[str, int],
                    body: Dict[str, Any],
                    parameter_constraints: List[str],
                    identifier: Optional[str]=None) -> 'RepetitionPulseTemplate':
        body = serializer.deserialize(body)
        return RepetitionPulseTemplate(body, repetition_count,
                                       identifier=identifier, parameter_constraints=parameter_constraints)


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
