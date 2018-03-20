"""This module defines SequencePulseTemplate, a higher-order hierarchical pulse template that
combines several other PulseTemplate objects for sequential execution."""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any, Iterable, Union, cast
from numbers import Real

from qctoolkit.serialization import Serializer

from qctoolkit.utils.types import MeasurementWindow, ChannelID, TimeType
from qctoolkit.pulses.pulse_template import PulseTemplate
from qctoolkit.pulses.parameters import Parameter, ParameterConstrainer
from qctoolkit.pulses.sequencing import InstructionBlock, Sequencer
from qctoolkit.pulses.conditions import Condition
from qctoolkit.pulses.pulse_template_parameter_mapping import \
    MissingMappingException, MappingPulseTemplate, MissingParameterDeclarationException, MappingTuple
from qctoolkit.pulses.instructions import Waveform
from qctoolkit.pulses.measurement import MeasurementDeclaration, MeasurementDefiner
from qctoolkit.expressions import Expression

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
                    yield from sub_waveform._sequenced_waveforms
                else:
                    yield sub_waveform

        self._sequenced_waveforms = tuple(flattened_sub_waveforms())
        self._duration = sum(waveform.duration for waveform in self._sequenced_waveforms)
        if not all(waveform.defined_channels == self.defined_channels for waveform in self._sequenced_waveforms[1:]):
            raise ValueError(
                "SequenceWaveform cannot be constructed from waveforms of different"
                "defined channels."
            )

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return self._sequenced_waveforms[0].defined_channels

    def unsafe_sample(self,
                      channel: ChannelID,
                      sample_times: np.ndarray,
                      output_array: Union[np.ndarray, None]=None) -> np.ndarray:
        if output_array is None:
            output_array = np.empty_like(sample_times)
        time = 0
        for subwaveform in self._sequenced_waveforms:
            # before you change anything here, make sure to understand the difference between basic and advanced
            # indexing in numpy and their copy/reference behaviour
            end = time + subwaveform.duration

            indices = slice(*np.searchsorted(sample_times, (float(time), float(end)), 'left'))
            subwaveform.unsafe_sample(channel=channel,
                                      sample_times=sample_times[indices]-time,
                                      output_array=output_array[indices])
            time = end
        return output_array

    @property
    def compare_key(self) -> Tuple[Waveform]:
        return self._sequenced_waveforms

    @property
    def duration(self) -> TimeType:
        return self._duration

    def get_measurement_windows(self) -> Iterable[MeasurementWindow]:
        def updated_measurement_window_generator(sequenced_waveforms):
            offset = 0
            for sub_waveform in sequenced_waveforms:
                for (name, begin, length) in sub_waveform.get_measurement_windows():
                    yield (name, begin+offset, length)
                offset += sub_waveform.duration
        return updated_measurement_window_generator(self._sequenced_waveforms)

    def unsafe_get_subset_for_channels(self, channels: Set[ChannelID]) -> 'Waveform':
        return SequenceWaveform(
            sub_waveform.unsafe_get_subset_for_channels(channels & sub_waveform.defined_channels)
            for sub_waveform in self._sequenced_waveforms if sub_waveform.defined_channels & channels)


class SequencePulseTemplate(PulseTemplate, ParameterConstrainer, MeasurementDefiner):
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

    def __init__(self,
                 *subtemplates: Union[PulseTemplate, MappingTuple],
                 external_parameters: Optional[Union[Iterable[str], Set[str]]]=None,
                 identifier: Optional[str]=None,
                 parameter_constraints: Optional[List[Union[str, Expression]]]=None,
                 measurements: Optional[List[MeasurementDeclaration]]=None) -> None:
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
        PulseTemplate.__init__(self, identifier=identifier)
        ParameterConstrainer.__init__(self, parameter_constraints=parameter_constraints)
        MeasurementDefiner.__init__(self, measurements=measurements)

        self.__subtemplates = [MappingPulseTemplate.from_tuple(st) if isinstance(st, tuple) else st
                               for st in subtemplates]

        # check that all subtemplates live on the same channels
        defined_channels = self.__subtemplates[0].defined_channels
        for subtemplate in self.__subtemplates[1:]:
            if subtemplate.defined_channels != defined_channels:
                raise ValueError('The subtemplates are defined for different channels')

        if external_parameters:
            external_parameters = set(external_parameters)
            remaining = external_parameters.copy()
            for subtemplate in self.__subtemplates:
                missing = subtemplate.parameter_names - external_parameters
                if missing:
                    raise MissingParameterDeclarationException(subtemplate, missing.pop())
                remaining -= subtemplate.parameter_names
            if not external_parameters >= self.constrained_parameters:
                raise MissingParameterDeclarationException(self,
                                                           (self.constrained_parameters-external_parameters).pop())
            remaining -= self.constrained_parameters
            if remaining:
                raise MissingMappingException(self, remaining.pop())

    @property
    def parameter_names(self) -> Set[str]:
        return set.union(*(st.parameter_names for st in self.__subtemplates))

    @property
    def subtemplates(self) -> List[MappingPulseTemplate]:
        return self.__subtemplates

    @property
    def is_interruptable(self) -> bool:
        return any(st.is_interruptable for st in self.subtemplates)

    @property
    def duration(self) -> Expression:
        return sum(sub.duration for sub in self.__subtemplates)

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return self.__subtemplates[0].defined_channels if self.__subtemplates else set()

    @property
    def measurement_names(self) -> Set[str]:
        return set.union(MeasurementDefiner.measurement_names.fget(self),
                         *(st.measurement_names for st in self.subtemplates))

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, 'Condition']) -> bool:
        """Returns the stop requirement of the first subtemplate. If a later subtemplate requires a stop the
        SequencePulseTemplate can be partially sequenced."""
        return self.__subtemplates[0].requires_stop(parameters, conditions) if self.__subtemplates else False

    def build_waveform(self,
                       parameters: Dict[str, Real],
                       channel_mapping: Dict[ChannelID, ChannelID]) -> SequenceWaveform:
        self.validate_parameter_constraints(parameters=parameters)
        return SequenceWaveform([sub_template.build_waveform(parameters,
                                                             channel_mapping=channel_mapping)
                                 for sub_template in self.__subtemplates])

    def build_sequence(self,
                       sequencer: Sequencer,
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, Condition],
                       measurement_mapping: Dict[str, str],
                       channel_mapping: Dict['ChannelID', 'ChannelID'],
                       instruction_block: InstructionBlock) -> None:
        self.validate_parameter_constraints(parameters=parameters)
        self.insert_measurement_instruction(instruction_block=instruction_block,
                                            parameters=parameters,
                                            measurement_mapping=measurement_mapping)
        for subtemplate in reversed(self.subtemplates):
            sequencer.push(subtemplate,
                           parameters=parameters,
                           conditions=conditions,
                           window_mapping=measurement_mapping,
                           channel_mapping=channel_mapping,
                           target_block=instruction_block)

    def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
        data = dict(subtemplates=[serializer.dictify(subtemplate) for subtemplate in self.subtemplates])
        if self.parameter_constraints:
            data['parameter_constraints'] = [str(c) for c in self.parameter_constraints]
        if self.measurement_declarations:
            data['measurements'] = self.measurement_declarations
        return data

    @staticmethod
    def deserialize(serializer: Serializer,
                    subtemplates: Iterable[Dict[str, Any]],
                    parameter_constraints: Optional[List[str]]=None,
                    identifier: Optional[str]=None,
                    measurements: Optional[List[MeasurementDeclaration]]=None) -> 'SequencePulseTemplate':
        subtemplates = [serializer.deserialize(st) for st in subtemplates]
        seq_template = SequencePulseTemplate(*subtemplates,
                                             parameter_constraints=parameter_constraints,
                                             identifier=identifier,
                                             measurements=measurements)
        return seq_template
