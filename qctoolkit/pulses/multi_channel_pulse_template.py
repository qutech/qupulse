"""This module defines MultiChannelPulseTemplate, which allows the combination of several
AtomicPulseTemplates into a single template spanning several channels.

Classes:
    - MultiChannelPulseTemplate: A pulse template defined for several channels by combining pulse
        templates
    - MultiChannelWaveform: A waveform defined for several channels by combining waveforms
"""

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
    """A MultiChannelWaveform is a Waveform object that allows combining arbitrary Waveform objects
    to into a single waveform defined for several channels.

    The number of channels used by the MultiChannelWaveform object is the sum of the channels used
    by the Waveform objects it consists of.

    MultiChannelWaveform allows an arbitrary mapping of channels defined by the Waveforms it
    consists of and the channels it defines. For example, if the MultiChannelWaveform consists
    of a two Waveform objects A and B which define two channels each, then the channels of the
    MultiChannelWaveform may be 0: A.1, 1: B.0, 2: B.1, 3: A.0 where A.0 means channel 0 of Waveform
    object A.

    The following constraints must hold:
     - The durations of all Waveform objects must be equal.
     - The channel mapping must be sane, i.e., no channel of the MultiChannelWaveform must be
        assigned more than one channel of any Waveform object it consists of
    """

    def __init__(self, subwaveforms: List[Tuple[Waveform, List[int]]]) -> None:
        """Create a new MultiChannelWaveform instance.

        Requires a list of subwaveforms in the form (Waveform, List(int)) where the list defines
        the channel mapping, i.e., a value y at index x in the list means that channel x of the
        subwaveform will be mapped to channel y of this MultiChannelWaveform object.

        Args:
            subwaveforms (List( (Waveform, List(int)) )): The list of subwaveforms of this
                MultiChannelWaveform as tuples of the form (Waveform, List(int))
        Raises:
            ValueError, if a channel mapping is out of bounds of the channels defined by this
                MultiChannelWaveform
            ValueError, if several subwaveform channels are assigned to a single channel of this
                MultiChannelWaveform
            ValueError, if subwaveforms have inconsistent durations
        """
        super().__init__()
        if not subwaveforms:
            raise ValueError(
                "MultiChannelWaveform cannot be constructed without channel waveforms."
            )

        self.__channel_waveforms = subwaveforms
        num_channels = self.num_channels
        duration = subwaveforms[0][0].duration
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
    """A multi-channel group of several AtomicPulseTemplate objects.

    While SequencePulseTemplate combines several subtemplates (with an identical number of channels)
    for subsequent execution, MultiChannelPulseTemplate combines several subtemplates into a new
    template defined for the sum of the channels of the subtemplates.
    A constraint is that the subtemplates must only be AtomicPulseTemplate objects, i.e., not define
    any changes in the control flow and be directly translatable into waveforms. Additionally,
    for each possible set of parameter value assignments, the waveforms resulting from the templates
    must be of the same duration. However, this cannot be enforced during the construction of the
    MultiChannelPulseTemplate object. Instead, if subtemplate misbehave, an exception will be raised
    during translation in the build_waveform and build_sequence methods.

    MultiChannelPulseTemplate allows an arbitrary mapping of channels defined by the subtemplates
    and the channels it defines. For example, if the MultiChannelPulseTemplate consists
    of a two subtemplates A and B which define two channels each, then the channels of the
    MultiChannelPulseTemplate may be 0: A.1, 1: B.0, 2: B.1, 3: A.0 where A.0 means channel 0 of A.
    The channel mapping must be sane, i.e., to no channel of the MultiChannelPulseTemplate must be
    assigned more than one channel of any subtemplate.

    Finally, MultiChannelPulseTemplate allows a mapping of parameter in the same way and with
    the same constraints as SequencePulseTemplate.

    See Also:
        - SequencePulseTemplate
        - AtomicPulseTemplate
        - MultiChannelWaveform
    """

    Subtemplate = Tuple[AtomicPulseTemplate, Dict[str, str], List[int]]

    def __init__(self,
                 subtemplates: Iterable[Subtemplate],
                 external_parameters: Set[str],
                 identifier: str=None) -> None:
        """Creates a new MultiChannelPulseTemplate instance.

        Requires a list of subtemplates in the form
        (PulseTemplate, Dict(str -> str), List(int)) where the dictionary is a mapping between the
        external parameters exposed by this SequencePulseTemplate to the parameters declared by the
        subtemplates, specifying how the latter are derived from the former, i.e., the mapping is
        subtemplate_parameter_name -> mapping_expression (as str) where the free variables in the
        mapping_expression are parameters declared by this MultiChannelPulseTemplate.
        The list defines the channel mapping, i.e., a value y at index x in the list means that
        channel x of the subtemplate will be mapped to channel y of this MultiChannelPulseTemplate.

        Args:
            subtemplates (List(Subtemplate)): The list of subtemplates of this
                MultiChannelPulseTemplate as tuples of the form
                (PulseTemplate, Dict(str -> str), List(int)).
            external_parameters (List(str)): A set of names for external parameters of this
                MultiChannelPulseTemplate.
            identifier (str): A unique identifier for use in serialization. (optional)
        Raises:
            ValueError, if a channel mapping is out of bounds of the channels defined by this
                MultiChannelPulseTemplate.
            ValueError, if several subtemplate channels are assigned to a single channel of this
                MultiChannelPulseTemplate.
            MissingMappingException, if a parameter of a subtemplate is not mapped to the external
                parameters of this MultiChannelPulseTemplate.
            MissingParameterDeclarationException, if a parameter mapping requires a parameter
                that was not declared in the external parameters of this MultiChannelPulseTemplate.
        """
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