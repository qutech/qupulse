"""This module defines MultiChannelPulseTemplate"""

from typing import Dict, List, Tuple, Set, Optional, Any, Iterable, Union

import numpy

from qctoolkit.serialization import Serializer
from qctoolkit.expressions import Expression

from qctoolkit.pulses.instructions import Waveform
from qctoolkit.pulses.pulse_template import AtomicPulseTemplate
from qctoolkit.pulses.sequence_pulse_template import PulseTemplateParameterMapping, \
    MissingMappingException
from qctoolkit.pulses.parameters import ParameterDeclaration, Parameter, \
    ParameterNotProvidedException
from qctoolkit.pulses.conditions import Condition
from qctoolkit.pulses.sequencing import Sequencer
from qctoolkit.pulses.instructions import InstructionBlock

__all__ = ["MultiChannelWaveform", "MultiChannelPulseTemplate"]


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
        assigned_channels = set()
        for waveform, channel_mapping in self.__channel_waveforms:
            if waveform.duration != duration:
                raise ValueError(
                    "MultiChannelWaveform cannot be constructed from channel waveforms of different"
                    "lengths."
                )
            # ensure that channel mappings stay within bounds
            out_of_bounds_channel = [channel for channel in channel_mapping
                                     if channel >= num_channels or channel < 0]
            if out_of_bounds_channel:
                raise ValueError(
                    "The channel mapping {}, assigned to a channel waveform, is not valid (must be "
                    "greater than 0 and less than {}).".format(out_of_bounds_channel.pop(),
                                                               num_channels)
                )
            # ensure that only a single waveform is mapped to each channel
            for channel in channel_mapping:
                if channel in assigned_channels:
                    raise ValueError("The channel {} has multiple channel waveform assignments"
                                     .format(channel))
                else:
                    assigned_channels.add(channel)


    @property
    def duration(self) -> float:
        return self.__channel_waveforms[0][0].duration

    def sample(self, sample_times: numpy.ndarray, first_offset: float=0) -> numpy.ndarray:
        voltages_transposed = numpy.empty((self.num_channels, len(sample_times)))
        for waveform, channel_mapping in self.__channel_waveforms:
            waveform_voltages_transposed = waveform.sample(sample_times, first_offset).T
            for old_c, new_c in enumerate(channel_mapping):
                voltages_transposed[new_c] = waveform_voltages_transposed[old_c]
        return voltages_transposed.T

    @property
    def num_channels(self) -> int:
        return sum([waveform.num_channels for waveform, _ in self.__channel_waveforms])

    @property
    def compare_key(self) -> Any:
        return self.__channel_waveforms


class MultiChannelPulseTemplate(AtomicPulseTemplate):

    Subtemplate = Tuple[AtomicPulseTemplate, Dict[str, str], List[int]]

    def __init__(self,
                 subtemplates: Iterable[Subtemplate],
                 external_parameters: Set[str],
                 identifier: str=None) -> None:
        super().__init__(identifier=identifier)
        self.__parameter_mapping = PulseTemplateParameterMapping(external_parameters)
        self.__subtemplates = [(template, channel_mapping) for (template, _, channel_mapping)
                               in subtemplates]

        assigned_channels = set()
        num_channels = self.num_channels
        for template, mapping_functions, channel_mapping in subtemplates:
            # Consistency checks
            for parameter, mapping_function in mapping_functions.items():
                self.__parameter_mapping.add(template, parameter, mapping_function)

            remaining = self.__parameter_mapping.get_remaining_mappings(template)
            if remaining:
                raise MissingMappingException(template,
                                              remaining.pop())


            # ensure that channel mappings stay within bounds
            out_of_bounds_channel = [channel for channel in channel_mapping
                                     if channel >= num_channels or channel < 0]
            if out_of_bounds_channel:
                raise ValueError(
                    "The channel mapping {}, assigned to a channel waveform, is not valid (must be "
                    "greater than 0 and less than {}).".format(out_of_bounds_channel.pop(),
                                                               num_channels)
                )
            # ensure that only a single waveform is mapped to each channel
            for channel in channel_mapping:
                if channel in assigned_channels:
                    raise ValueError("The channel {} has multiple channel waveform assignments"
                                     .format(channel))
                else:
                    assigned_channels.add(channel)

    @property
    def parameter_names(self) -> Set[str]:
        return self.__parameter_mapping.external_parameters

    @property
    def parameter_declarations(self) -> Set[ParameterDeclaration]:
        # TODO: min, max, default values not mapped (required?)
        return {ParameterDeclaration(parameter_name) for parameter_name in self.parameter_names}

    def get_measurement_windows(self, parameters: Dict[str, Parameter] = None) \
            -> List['MeasurementWindow']:
        raise NotImplementedError()

    @property
    def is_interruptable(self) -> bool:
        return all(t.is_interruptable for t, _ in self.__subtemplates)

    @property
    def num_channels(self) -> int:
        return sum(t.num_channels for t, _ in self.__subtemplates)

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, Condition]) -> bool:
        return any(t.requires_stop(parameters, conditions) for t, _ in self.__subtemplates)

    def build_waveform(self, parameters: Dict[str, Parameter]) -> Optional[Waveform]:
        missing = self.parameter_names - parameters.keys()
        if missing:
            raise ParameterNotProvidedException(missing.pop())

        channel_waveforms = []
        for template, channel_mapping in self.__subtemplates:
            inner_parameters = self.__parameter_mapping.map_parameters(template, parameters)
            waveform = template.build_waveform(inner_parameters)
            channel_waveforms.append((waveform, channel_mapping))
        waveform = MultiChannelWaveform(channel_waveforms)
        return waveform

    def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
        subtemplates = []
        for subtemplate, channel_mapping in self.__subtemplates:
            mapping_functions = self.__parameter_mapping.get_template_map(subtemplate)
            mapping_function_strings = \
                {k: serializer.dictify(m) for k, m in mapping_functions.items()}
            subtemplate = serializer.dictify(subtemplate)
            subtemplates.append(dict(template=subtemplate,
                                     parameter_mappings=mapping_function_strings,
                                     channel_mappings=channel_mapping))
        return dict(subtemplates=subtemplates,
                    external_parameters=sorted(list(self.parameter_names)))

    @staticmethod
    def deserialize(serializer: Serializer,
                    subtemplates: Iterable[Dict[str, Any]],
                    external_parameters: Iterable[str],
                    identifier: Optional[str]=None) -> 'MultiChannelPulseTemplate':
        subtemplates = \
            [(serializer.deserialize(subt['template']),
              {k: str(serializer.deserialize(m)) for k, m in subt['parameter_mappings'].items()},
              subt['channel_mappings'])
             for subt in subtemplates]

        template = MultiChannelPulseTemplate(subtemplates,
                                             external_parameters,
                                             identifier=identifier)
        return template