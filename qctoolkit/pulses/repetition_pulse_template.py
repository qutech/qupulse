"""This module defines RepetitionPulseTemplate, a higher-order hierarchical pulse template that
represents the n-times repetition of another PulseTemplate."""

from typing import Dict, List, Set, Optional, Union, Any, Iterable, Tuple

import numpy as np

from qctoolkit.serialization import Serializer

from qctoolkit import MeasurementWindow, ChannelID
from qctoolkit.pulses.pulse_template import PulseTemplate, PossiblyAtomicPulseTemplate
from qctoolkit.pulses.loop_pulse_template import LoopPulseTemplate
from qctoolkit.pulses.sequencing import Sequencer
from qctoolkit.pulses.instructions import InstructionBlock, InstructionPointer, Waveform
from qctoolkit.pulses.parameters import ParameterDeclaration, Parameter, ConstantParameter
from qctoolkit.pulses.conditions import Condition


__all__ = ["RepetitionPulseTemplate", "ParameterNotIntegerException"]


class RepetitionWaveform(Waveform):
    """This class allows putting multiple PulseTemplate together in one waveform on the hardware."""
    def __init__(self, body: Waveform, repetition_count: int):
        """

        :param subwaveforms: All waveforms must have the same defined channels
        """
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


class RepetitionPulseTemplate(LoopPulseTemplate):
    """Repeat a PulseTemplate a constant number of times.

    The equivalent to a simple for-loop in common programming languages in qctoolkit's pulse
    modelling.
    """

    def __init__(self,
                 body: PulseTemplate,
                 repetition_count: Union[int, ParameterDeclaration, str],
                 identifier: Optional[str]=None) -> None:
        """Create a new RepetitionPulseTemplate instance.

        Args:
            body (PulseTemplate): The PulseTemplate which will be repeated.
            repetition_count (int or ParameterDeclaration): The number of repetitions either as a
                constant integer value or as a parameter declaration.
            loop_index (str): If specified the loop index
            identifier (str): A unique identifier for use in serialization. (optional)
        """
        super().__init__(identifier=identifier, body=body)
        if isinstance(repetition_count, float) and repetition_count.is_integer():
            repetition_count = int(repetition_count)

        if isinstance(repetition_count, str):
            self._repetition_count = ParameterDeclaration(repetition_count, min=0)
        elif isinstance(repetition_count, (int, ParameterDeclaration)):
            self._repetition_count = repetition_count
        else:
            raise ValueError('Invalid repetition count type: {}'.format(type(repetition_count)))

    @property
    def repetition_count(self) -> ParameterDeclaration:
        """The amount of repetitions. Either a constant integer or a ParameterDeclaration object."""
        return self._repetition_count

    def get_repetition_count_value(self, parameters: Dict[str, Parameter]) -> int:
        if isinstance(self._repetition_count, ParameterDeclaration):
            value = self._repetition_count.get_value(parameters)
            if isinstance(value, float) and not value.is_integer():
                raise ParameterNotIntegerException(self._repetition_count.name, value)
            return int(value)
        else: return self._repetition_count

    def __str__(self) -> str:
        return "RepetitionPulseTemplate: <{}> times <{}>"\
            .format(self._repetition_count, self.body)

    @property
    def parameter_names(self) -> Set[str]:
        return set(parameter.name for parameter in self.parameter_declarations)

    @property
    def parameter_declarations(self) -> Set[str]:
        return self.body.parameter_declarations | ({self._repetition_count} if isinstance(self._repetition_count,
                                                                                          ParameterDeclaration)
                                                   else set())

    @property
    def measurement_names(self) -> Set[str]:
        return self.body.measurement_names

    def build_waveform(self, parameters: Dict[str, Parameter]) -> RepetitionWaveform:
        return RepetitionWaveform(self.body.build_waveform(parameters), self.get_repetition_count_value(parameters))

    def build_sequence(self,
                       sequencer: Sequencer,
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, Condition],
                       measurement_mapping: Dict[str, str],
                       channel_mapping: Dict['ChannelID', 'ChannelID'],
                       instruction_block: InstructionBlock) -> None:

        if self.atomicity:
            # atomicity can only be enabled if the loop index is not used
            self.atomic_build_sequence(parameters=parameters,
                                       measurement_mapping=measurement_mapping,
                                       channel_mapping=channel_mapping,
                                       instruction_block=instruction_block)
        else:
            body_block = InstructionBlock()
            body_block.return_ip = InstructionPointer(instruction_block, len(instruction_block))

            instruction_block.add_instruction_repj(self.get_repetition_count_value(parameters), body_block)
            sequencer.push(self.body, parameters, conditions, measurement_mapping, channel_mapping, body_block)

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, Condition]) -> bool:
        if isinstance(self._repetition_count, ParameterDeclaration):
            return parameters[self._repetition_count.name].requires_stop
        else:
            return False

    def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
        repetition_count = self._repetition_count
        if isinstance(repetition_count, ParameterDeclaration):
            repetition_count = serializer.dictify(repetition_count)
        return dict(
            type=serializer.get_type_identifier(self),
            body=serializer.dictify(self.body),
            repetition_count=repetition_count,
            atomicity=self.atomicity,
        )

    @staticmethod
    def deserialize(serializer: Serializer,
                    repetition_count: Dict[str, Any],
                    body: Dict[str, Any],
                    atomicity: bool,
                    identifier: Optional[str]=None) -> 'Serializable':
        body = serializer.deserialize(body)
        if isinstance(repetition_count, dict):
            repetition_count = serializer.deserialize(repetition_count)
        result = RepetitionPulseTemplate(body, repetition_count, identifier=identifier)
        result.atomicity = atomicity
        return result


class ParameterNotIntegerException(Exception):
    """Indicates that the value of the parameter given as repetition count was not an integer."""

    def __init__(self, parameter_name: str, parameter_value: float) -> None:
        super().__init__()
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value

    def __str__(self) -> str:
        return "The parameter <{}> must have an integral value (was <{}>) " \
            "for use as repetition count.".format(
                self.parameter_name,
                self.parameter_value
            )
