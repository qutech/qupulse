"""This module defines SequencePulseTemplate, a higher-order hierarchical pulse template that
combines several other PulseTemplate objects for sequential execution."""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any, Iterable, Union
import itertools

from qctoolkit.serialization import Serializer

from qctoolkit.pulses.pulse_template import PulseTemplate, MeasurementWindow, PossiblyAtomicPulseTemplate
from qctoolkit.pulses.parameters import ParameterDeclaration, Parameter
from qctoolkit.pulses.sequencing import InstructionBlock, Sequencer
from qctoolkit.pulses.conditions import Condition
from qctoolkit.pulses.pulse_template_parameter_mapping import \
    MissingMappingException, MappingTemplate, ChannelID, MissingParameterDeclarationException
from qctoolkit.pulses.instructions import Waveform

__all__ = ["SequencePulseTemplate"]


class SequenceWaveform(Waveform):
    """This class allows putting multiple PulseTemplate together in one waveform on the hardware."""
    def __init__(self, subwaveforms: List[Waveform]):
        """

        :param subwaveforms: All waveforms must have the same defined channels
        """
        if not subwaveforms:
            raise ValueError(
                "SequenceWaveform cannot be constructed without channel waveforms."
            )

        def flattened_sub_waveforms():
            for sub_waveform in subwaveforms:
                if isinstance(sub_waveform, SequenceWaveform):
                    yield from sub_waveform.__sequenced_waveforms
                else:
                    yield sub_waveform

        self.__sequenced_waveforms = tuple(flattened_sub_waveforms())
        self.__duration = sum(waveform.duration for waveform in self.__sequenced_waveforms)
        if not all(waveform.defined_channels == self.defined_channels for waveform in self.__sequenced_waveforms[1:]):
            raise ValueError(
                "SequenceWaveform cannot be constructed from waveforms of different"
                "defined channels."
            )

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return self.__sequenced_waveforms[0].defined_channels

    def unsafe_sample(self,
                      channel: ChannelID,
                      sample_times: np.ndarray,
                      output_array: Union[np.ndarray, None]=None):
        if output_array is None:
            output_array = np.empty(len(sample_times))
        time = 0
        for subwaveform in self.__sequenced_waveforms:
            # before you change anything here, make sure to understand the difference between basic and advanced
            # indexing in numpy and their copy/reference behaviour
            indices = slice(*np.searchsorted(sample_times, (time, time+subwaveform.duration), 'left'))
            subwaveform.unsafe_sample(channel=channel,
                                      sample_times=sample_times[indices],
                                      output_array=output_array[indices])
            time += subwaveform.duration
        return output_array

    @property
    def compare_key(self):
        return self.__sequenced_waveforms

    @property
    def duration(self):
        return self.__duration

    def get_measurement_windows(self) -> Iterable[MeasurementWindow]:
        def updated_measurement_window_generator(sequenced_waveforms):
            offset = 0
            for sub_waveform in sequenced_waveforms:
                for (name, begin, length) in sub_waveform.get_measurement_windows():
                    yield (name, begin+offset, length)
                offset += sub_waveform.duration
        return updated_measurement_window_generator(self.__sequenced_waveforms)

    def unsafe_get_subset_for_channels(self, channels: Set[ChannelID]) -> 'Waveform':
        return SequenceWaveform(
            sub_waveform.unsafe_get_subset_for_channels(channels & sub_waveform.defined_channels)
            for sub_waveform in self.__sequenced_waveforms if sub_waveform.defined_channels & channels)


class SequencePulseTemplate(PossiblyAtomicPulseTemplate):
    """A sequence of different PulseTemplates.
    
    SequencePulseTemplate allows to group several
    PulseTemplates (subtemplates) into one larger sequence,
    i.e., when instantiating a pulse from a SequencePulseTemplate
    all pulses instantiated from the subtemplates are queued for
    execution right after one another.
    SequencePulseTemplate requires to specify a mapping of
    parameter declarations from its subtemplates to its own, enabling
    renaming and mathematical transformation of parameters.
    """

    # a subtemplate consists of a pulse template and mapping functions for its "internal" parameters
    SimpleSubTemplate = Tuple[PulseTemplate, Dict[str, str]]  # pylint: disable=invalid-name

    def __init__(self,
                 subtemplates: Iterable[Union[SimpleSubTemplate, MappingTemplate]],
                 external_parameters: Union[Iterable[str], Set[str]],  # pylint: disable=invalid-sequence-index
                 identifier: Optional[str]=None) -> None:
        """Create a new SequencePulseTemplate instance.

        Requires a (correctly ordered) list of subtemplates in the form
        (PulseTemplate, Dict(str -> str)) where the dictionary is a mapping between the external
        parameters exposed by this SequencePulseTemplate to the parameters declared by the
        subtemplates, specifying how the latter are derived from the former, i.e., the mapping is
        subtemplate_parameter_name -> mapping_expression (as str) where the free variables in the
        mapping_expression are parameters declared by this SequencePulseTemplate.

        The following requirements must be satisfied:
            - for each parameter declared by a subtemplate, a mapping expression must be provided
            - each free variable in a mapping expression must be declared as an external parameter
                of this SequencePulseTemplate

        Args:
            subtemplates (List(Subtemplate)): The list of subtemplates of this
                SequencePulseTemplate as tuples of the form (PulseTemplate, Dict(str -> str)).
            external_parameters (List(str)): A set of names for external parameters of this
                SequencePulseTemplate.
            identifier (str): A unique identifier for use in serialization. (optional)
        Raises:
            MissingMappingException, if a parameter of a subtemplate is not mapped to the external
                parameters of this SequencePulseTemplate.
            MissingParameterDeclarationException, if a parameter mapping requires a parameter
                that was not declared in the external parameters of this SequencePulseTemplate.
        """
        super().__init__(identifier)

        self.__subtemplates = [st if not isinstance(st, tuple) else MappingTemplate(*st) for st in subtemplates]
        external_parameters = external_parameters if isinstance(external_parameters,set) else set(external_parameters)

        # check that all subtempaltes live on the same channels
        defined_channels = self.__subtemplates[0].defined_channels
        for subtemplate in self.__subtemplates[1:]:
            if subtemplate.defined_channels != defined_channels:
                raise ValueError('The subtemplates are defined for different channels')

        remaining = external_parameters.copy()
        for subtemplate in self.__subtemplates:
            missing = subtemplate.parameter_names - external_parameters
            if missing:
                raise MissingParameterDeclarationException(subtemplate.template,missing.pop())
            remaining = remaining - subtemplate.parameter_names
        if remaining:
            MissingMappingException(subtemplate.template,remaining.pop())
        self.__atomicity = False

    @property
    def parameter_names(self) -> Set[str]:
        return set.union(*(st.parameter_names for st in self.__subtemplates))

    @property
    def parameter_declarations(self) -> Set[ParameterDeclaration]:
        # TODO: min, max, default values not mapped (required?)
        return {ParameterDeclaration(name) for name in self.parameter_names}

    @property
    def subtemplates(self) -> List[MappingTemplate]:
        return self.__subtemplates

    @property
    def is_interruptable(self) -> bool:
        return any(st.is_interruptable for st in self.subtemplates)

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return self.__subtemplates[0].defined_channels if self.__subtemplates else set()

    @property
    def measurement_names(self) -> Set[str]:
        return set.union(*(st.measurement_names for st in self.subtemplates))

    @property
    def atomicity(self) -> bool:
        if any(subtemplate.atomicity is False for subtemplate in self.__subtemplates):
            self.__atomicity = False
        return self.__atomicity

    @atomicity.setter
    def atomicity(self, val: bool) -> None:
        if val and any(subtemplate.atomicity is False for subtemplate in self.__subtemplates):
            raise ValueError('Cannot make atomic as not all sub templates are atomic')
        self.__atomicity = val

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, 'Condition']) -> bool:
        """Returns the stop requirement of the first subtemplate. If a later subtemplate requires a stop the
        SequencePulseTemplate can be partially sequenced."""
        return self.__subtemplates[0].requires_stop(parameters,conditions) if self.__subtemplates else False

    def build_waveform(self, parameters: Dict[str, Parameter]):
        return SequenceWaveform([subtemplate.build_waveform(parameters) for subtemplate in self.__subtemplates])

    def build_sequence(self,
                       sequencer: Sequencer,
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, Condition],
                       measurement_mapping: Dict[str, str],
                       channel_mapping: Dict['ChannelID', 'ChannelID'],
                       instruction_block: InstructionBlock) -> None:
        # todo: currently ignores is_interruptable
        if self.atomicity:
            self.atomic_build_sequence(parameters=parameters,
                                       measurement_mapping=measurement_mapping,
                                       channel_mapping=channel_mapping,
                                       instruction_block=instruction_block)
        else:
            for subtemplate in reversed(self.subtemplates):
                sequencer.push(subtemplate, parameters, conditions, measurement_mapping, channel_mapping,
                               instruction_block)

    def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
        data = dict()

        data['subtemplates'] = [serializer.dictify(subtemplate) for subtemplate in self.subtemplates]
        data['type'] = serializer.get_type_identifier(self)

        return data

    @staticmethod
    def deserialize(serializer: Serializer,
                    subtemplates: Iterable[Dict[str, Any]],
                    identifier: Optional[str]=None) -> 'SequencePulseTemplate':
        subtemplates = [serializer.deserialize(st) for st in subtemplates]
        external_parameters = set.union( *(st.parameter_names for st in subtemplates) )
        seq_template = SequencePulseTemplate(subtemplates, external_parameters, identifier=identifier)
        return seq_template
