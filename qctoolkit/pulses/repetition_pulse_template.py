"""This module defines RepetitionPulseTemplate, a higher-order hierarchical pulse template that
represents the n-times repetition of another PulseTemplate."""

from typing import Dict, List, Set, Optional, Union, Any

import numpy as np

from qctoolkit.serialization import Serializer

from qctoolkit.pulses.pulse_template import PulseTemplate, ChannelID, PossiblyAtomicPulseTemplate
from qctoolkit.pulses.sequencing import Sequencer
from qctoolkit.pulses.instructions import InstructionBlock, InstructionPointer, Waveform
from qctoolkit.pulses.parameters import ParameterDeclaration, Parameter
from qctoolkit.pulses.conditions import Condition


__all__ = ["RepetitionPulseTemplate", "ParameterNotIntegerException"]


class RepetitionWaveform(Waveform):
    """This class allows putting multiple PulseTemplate together in one waveform on the hardware."""
    def __init__(self, body: Waveform, repetition_count: int):
        """

        :param subwaveforms: All waveforms must have the same defined channels
        """
        self.__body = body
        self.__repetition_count = repetition_count
        if repetition_count < 1 or not isinstance(repetition_count, int):
            raise ValueError('Repetition count must be an integer >0')

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return self.__body.defined_channels

    def unsafe_sample(self,
                      channel: ChannelID,
                      sample_times: np.ndarray,
                      output_array: Union[np.ndarray, None]=None):
        if output_array is None:
            output_array = np.empty(len(sample_times))
        body_duration = self.__body.duration
        for i in range(self.__repetition_count):
            indices = slice(*np.searchsorted(sample_times, (i*body_duration, (i+1)*body_duration)))
            self.__body.unsafe_sample(channel=channel,
                                      sample_times=sample_times[indices],
                                      output_array=output_array[indices])
        return output_array

    @property
    def compare_key(self):
        return self.__body.compare_key, self.__repetition_count

    @property
    def duration(self):
        return self.__body.duration*self.__repetition_count

    def get_measurement_windows(self):
        def get_measurement_window_generator(body: Waveform, repetition_count: int):
            body_windows = list(body.get_measurement_windows())
            for i in range(repetition_count):
                for (name, begin, length) in body_windows:
                    yield (name, begin+i*body.duration, length)
        return get_measurement_window_generator(self.__body, self.__repetition_count)

    def unsafe_get_subset_for_channels(self, channels: Set[ChannelID]):
        return RepetitionWaveform(body=self.__body.unsafe_get_subset_for_channels(channels),
                                  repetition_count=self.__repetition_count)


class RepetitionPulseTemplate(PossiblyAtomicPulseTemplate):
    """Repeat a PulseTemplate a constant number of times.

    The equivalent to a simple for-loop in common programming languages in qctoolkit's pulse
    modelling.
    """

    def __init__(self,
                 body: PulseTemplate,
                 repetition_count: Union[int, ParameterDeclaration],
                 identifier: Optional[str]=None) -> None:
        """Create a new RepetitionPulseTemplate instance.

        Args:
            body (PulseTemplate): The PulseTemplate which will be repeated.
            repetition_count (int or ParameterDeclaration): The number of repetitions either as a
                constant integer value or as a parameter declaration.
            identifier (str): A unique identifier for use in serialization. (optional)
        """
        super().__init__(identifier=identifier)
        self.__body = body
        self.__repetition_count = repetition_count
        self.__atomicity = False

    @property
    def body(self) -> PulseTemplate:
        """The PulseTemplate which is repeated by this RepetitionPulseTemplate."""
        return self.__body

    @property
    def repetition_count(self) -> Union[int, ParameterDeclaration]:
        """The amount of repetitions. Either a constant integer or a ParameterDeclaration object."""
        return self.__repetition_count

    def __str__(self) -> str:
        return "RepetitionPulseTemplate: <{}> times <{}>"\
            .format(self.__repetition_count, self.__body)

    @property
    def parameter_names(self) -> Set[str]:
        return self.__body.parameter_names

    @property
    def parameter_declarations(self) -> Set[str]:
        return self.__body.parameter_declarations

    @property
    def is_interruptable(self) -> bool:
        return self.__body.is_interruptable

    @property
    def defined_channels(self) -> Set['ChannelID']:
        return self.__body.defined_channels

    @property
    def measurement_names(self) -> Set[str]:
        return self.__body.measurement_names

    @property
    def atomicity(self) -> bool:
        if self.__body.atomicity is False:
            self.__atomicity = False
        return self.__atomicity

    @atomicity.setter
    def atomicity(self, val) -> None:
        if val and self.__body.atomicity is False:
            raise ValueError('Cannot make RepetitionPulseTemplate atomic as the body is not')
        self.__atomicity = val

    def build_waveform(self, parameters: Dict[str, Parameter]) -> RepetitionWaveform:
        return RepetitionWaveform(self.__body.build_waveform(parameters),
                                  self.__repetition_count.get_value(parameters))

    def build_sequence(self,
                       sequencer: Sequencer,
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, Condition],
                       measurement_mapping: Dict[str, str],
                       channel_mapping: Dict['ChannelID', 'ChannelID'],
                       instruction_block: InstructionBlock) -> None:
        if self.atomicity:
            self.atomic_build_sequence(parameters=parameters,
                                       measurement_mapping=measurement_mapping,
                                       channel_mapping=channel_mapping,
                                       instruction_block=instruction_block)
        else:
            repetition_count = self.__repetition_count
            if isinstance(repetition_count, ParameterDeclaration):
                repetition_count = repetition_count.get_value(parameters)
                if not repetition_count.is_integer():
                    raise ParameterNotIntegerException(self.__repetition_count.name, repetition_count)

            body_block = InstructionBlock()
            body_block.return_ip = InstructionPointer(instruction_block, len(instruction_block))

            instruction_block.add_instruction_repj(int(repetition_count), body_block)
            sequencer.push(self.body, parameters, conditions, measurement_mapping, channel_mapping, body_block)

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, Condition]) -> bool:
        if isinstance(self.__repetition_count, ParameterDeclaration):
            if parameters[self.__repetition_count.name].requires_stop:
                return True
        return False

    def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
        repetition_count = self.__repetition_count
        if isinstance(repetition_count, ParameterDeclaration):
            repetition_count = serializer.dictify(repetition_count)
        return dict(
            type=serializer.get_type_identifier(self),
            body=serializer.dictify(self.__body),
            repetition_count=repetition_count,
            atomicity=self.atomicity
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
