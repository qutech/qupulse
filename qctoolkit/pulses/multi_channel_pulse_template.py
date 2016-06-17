"""This module defines MultiChannelPulseTemplate"""

from typing import Dict, List, Tuple, Set, Optional, Any, Iterable, Union

from qctoolkit.serialization import Serializer
from qctoolkit.expressions import Expression

from qctoolkit.pulses.instructions import Waveform
from qctoolkit.pulses.pulse_template import PulseTemplate, AtomicPulseTemplate
from qctoolkit.pulses.sequence_pulse_template import PulseTemplateParameterMapping, \
    MissingMappingException

__all__ = ["MultiChannelPulseTemplate"]


class MultiChannelWaveform(Waveform):

    def __init__(self, channel_waveforms: List[Tuple[Waveform, List[int]]]) -> None:
        super().__init__()
        if not channel_waveforms:
            raise ValueError(
                "MultiChannelWaveform cannot be constructed without channel waveforms."
            )

        self.__channel_waveforms = channel_waveforms
        num_channels = self.num_channels
        duration = channel_waveforms[0][0].duration
        for waveform, channel_mapping in self.__channel_waveforms:
            if waveform.duration != duration:
                raise ValueError(
                    "MultiChannelWaveform cannot be constructed from channel waveforms of different"
                    "lengths."
                )
            too_high_channel = [channel for channel in channel_mapping if channel >= num_channels]
            if too_high_channel:
                raise ValueError(
                    "The channel {}, assigned to a channel waveform, does exceed the number of"
                    "channels {}.".format(too_high_channel.pop(), num_channels)
                )

    @property
    def duration(self) -> float:
        return channel_waveforms[0][0].duration

    def sample(self, sample_times: numpy.ndarray, first_offset: float=0) -> numpy.ndarray:
        voltages = np.empty((len(sample_times), self.num_channels))
        for waveform, channel_mapping in self.__channel_waveforms:
            waveform_voltages = waveform.sample(sample_times, first_offset)
            for old_c, new_c in enumerate(channel_mapping):
                voltages[new_c] = waveform_voltages[old_c]
        return voltages

    @property
    def num_channels(self) -> int:
        return sum([waveform.num_channels for waveform, _ in self.__channel_waveforms])


Subtemplate = Tuple[AtomicPulseTemplate, Dict[str, str], List[int]]


class MultiChannelPulseTemplate(PulseTemplate):

    def __init__(self,
                 subtemplates: Iterable[Subtemplate],
                 external_parameters: Set[str],
                 identifier: str=None) -> None:
        super().__init__(identifier=identifier)
        self.__parameter_mapping = PulseTemplateParameterMapping(external_parameters)

        #todo: check duration consistency

        for template, mapping_functions in subtemplates:
            # Consistency checks
            for parameter, mapping_function in mapping_functions.items():
                self.__parameter_mapping.add(template, parameter, mapping_function)

            remaining = self.__parameter_mapping.get_remaining_mappings(template)
            if remaining:
                raise MissingMappingException(template,
                                              remaining.pop())

        self.__subtemplates = [(template, channel_mapping) for (template, _, channel_mapping)
                               in subtemplates]

    @property
    def parameter_names(self) -> Set[str]:
        return self.__parameter_mapping.external_parameters

    @property
    def parameter_declarations(self) -> Set[ParameterDeclaration]:
        # TODO: min, max, default values not mapped (required?)
        return {ParameterDeclaration(parameter_name) for parameter_name in self.parameter_names}

    def get_measurement_windows(self, parameters: Dict[str, Parameter] = None) \
            -> List[MeasurementWindow]:
        raise NotImplementedError()

    @property
    def is_interruptable(self) -> bool:
        return all(self.__subtemplates.is_interruptable)

    @property
    def num_channels(self) -> int:
        return sum(self.__subtemplates.num_channels)

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, Condition]) -> bool:
        return any(self.__subtemplates.requires_stop(parameters, conditions))

    def build_sequence(self,
                       sequencer: Sequencer,
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, Condition],
                       instruction_block: InstructionBlock) -> None:
        missing = self.parameter_names - parameters.keys()
        if missing:
            raise ParameterNotProvidedException(missing[0])

        channel_waveforms = []
        for template, channel_mapping in self.__subtemplates:
            inner_parameters = self.__parameter_mapping.map_parameters(template, parameters)
            waveform = template.build_waveform(inner_parameters)
            channel_waveforms.append((waveform, channel_mapping))
        waveform = MultiChannelWaveform(channel_waveforms)
        instruction_block.add_instruction_exec(waveform)

    def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
        raise NotImplementedError()

    @staticmethod
    def deserialize(serializer: Serializer, **kwargs) -> 'MultiChannelPulseTemplate':
        raise NotImplementedError()