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

from qctoolkit.pulses.instructions import InstructionBlock, SingleChannelWaveform, InstructionPointer
from qctoolkit.pulses.pulse_template import PulseTemplate
from qctoolkit.pulses.pulse_template_parameter_mapping import MissingMappingException, MappingTemplate,\
    MissingParameterDeclarationException, ChannelID
from qctoolkit.pulses.parameters import ParameterDeclaration, Parameter, \
    ParameterNotProvidedException
from qctoolkit.pulses.conditions import Condition
from qctoolkit.comparable import Comparable

__all__ = ["MultiChannelWaveform", "MultiChannelPulseTemplate"]


class MultiChannelWaveform(Comparable):
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

    def __init__(self, subwaveforms: Dict[ChannelID, SingleChannelWaveform]) -> None:
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
        duration = next(iter(self.__channel_waveforms.values())).duration
        if not all(waveform.duration == duration for waveform in self.__channel_waveforms.values()):
            raise ValueError(
                "MultiChannelWaveform cannot be constructed from channel waveforms of different"
                "lengths."
            )

    @property
    def duration(self) -> float:
        return next(iter(self.__channel_waveforms.values())).duration

    def __getitem__(self, key: Union[Set[ChannelID], ChannelID]) -> Union[SingleChannelWaveform, 'MultiChannelWaveform']:
        try:
            if not isinstance(key, (set, frozenset)):
                return self.__channel_waveforms[key]
            else:
                return MultiChannelWaveform(dict((chID, self.__channel_waveforms[chID]) for chID in key))
        except KeyError as err:
            raise KeyError('Unknown channel ID: {}'.format(err.args[0]),*err.args)

    def __add__(self, other: 'MultiChannelWaveform') -> 'MultiChannelWaveform':
        if set(self.__channel_waveforms) | set(other.__channel_waveforms):
            raise ChannelMappingException(set(self.__channel_waveforms) | set(other.__channel_waveforms))
        return MultiChannelWaveform(dict(**self.__channel_waveforms,**other.__channel_waveforms))

    def __iadd__(self, other: 'MultiChannelWaveform'):
        if set(self.__channel_waveforms) | set(other.__channel_waveforms):
            raise ChannelMappingException(set(self.__channel_waveforms) | set(other.__channel_waveforms))
        if self.duration != other.duration:
            raise ValueError('Waveforms not combinable as they have different durations.')
        self.__channel_waveforms = dict(**self.__channel_waveforms,**other.__channel_waveforms)

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return set(self.__channel_waveforms.keys())

    @property
    def compare_key(self) -> Any:
        return self.__channel_waveforms

    def get_remapped(self, channel_mapping):
        return MultiChannelWaveform(
            {channel_mapping[old_channel]: waveform for old_channel, waveform in self.__channel_waveforms.items()}
        )


class MultiChannelPulseTemplate(PulseTemplate):
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

    SimpleSubTemplate = Tuple[PulseTemplate, Dict[str, str], Dict[ChannelID, ChannelID]]

    def __init__(self,
                 subtemplates: Iterable[Union[PulseTemplate, SimpleSubTemplate]],
                 external_parameters: Set[str],
                 identifier: str=None) -> None:
        """Creates a new MultiChannelPulseTemplate instance.

        Requires a list of subtemplates in the form
        (PulseTemplate, Dict(str -> str), List(int)) where the dictionary is a mapping between the
        external parameters exposed by this PulseTemplate to the parameters declared by the
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
        super().__init__(identifier)

        def to_mapping_template(template, parameter_mapping, channel_mapping):
            if not (isinstance(channel_mapping, dict) and all(
                        isinstance(ch1,(int,str)) and isinstance(ch2,(int,str)) for ch1, ch2 in channel_mapping.items())):
                raise ValueError('{} is not a valid channel mapping.'.format(channel_mapping))
            return MappingTemplate(template, parameter_mapping, channel_mapping=channel_mapping)
        self.__subtemplates = [st if isinstance(st, PulseTemplate) else to_mapping_template(*st) for st in subtemplates]

        defined_channels = [st.defined_channels for st in self.__subtemplates]
        # check there are no intersections between channels
        for i, chans1 in enumerate(defined_channels):
            for j, chans2 in enumerate(defined_channels[i+1:]):
                if chans1 & chans2:
                    raise ChannelMappingException(self.__subtemplates[i],
                                                  self.__subtemplates[i+1+j],
                                                  (chans1 | chans2).pop())

        remaining = external_parameters.copy()
        for subtemplate in self.__subtemplates:
            missing = subtemplate.parameter_names - external_parameters
            if missing:
                raise MissingParameterDeclarationException(subtemplate.template, missing.pop())
            remaining -= subtemplate.parameter_names
        if remaining:
            raise MissingMappingException(subtemplate.template, remaining.pop())

    @property
    def parameter_names(self) -> Set[str]:
        return set.union(*(st.parameter_names for st in self.__subtemplates))

    @property
    def parameter_declarations(self) -> Set[ParameterDeclaration]:
        # TODO: min, max, default values not mapped (required?)
        return {ParameterDeclaration(name) for name in self.parameter_names}

    @property
    def subtemplates(self) -> Iterable[MappingTemplate]:
        return iter(self.__subtemplates)

    @property
    def is_interruptable(self) -> bool:
        return all(st.is_interruptable for st in self.subtemplates)

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return set.union(*(st.defined_channels for st in self.__subtemplates))

    @property
    def measurement_names(self) -> Set[str]:
        return set.union(*(st.measurement_names for st in self.__subtemplates))

    def build_sequence(self,
                       sequencer: 'Sequencer',
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, 'Condition'],
                       measurement_mapping: Dict[str, str],
                       channel_mapping: Dict['ChannelID', 'ChannelID'],
                       instruction_block: InstructionBlock) -> None:
        channel_to_instruction_block = dict()
        for subtemplate in self.subtemplates:
            block = InstructionBlock()
            sequencer.push(subtemplate, parameters, conditions, measurement_mapping, channel_mapping, block)
            channel_to_instruction_block.update(dict.fromkeys(subtemplate.defined_channels, block))
        instruction_block.add_instruction_chan(channel_to_instruction=channel_to_instruction_block)

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, 'Condition']) -> bool:
        return any(st.requires_stop(parameters, conditions) for st in self.__subtemplates)


    def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
        data = dict(subtemplates=[serializer.dictify(subtemplate) for subtemplate in self.subtemplates],
                    type=serializer.get_type_identifier(self))
        return data

    @staticmethod
    def deserialize(serializer: Serializer,
                    subtemplates: Iterable[Dict[str, Any]],
                    identifier: Optional[str]=None) -> 'MultiChannelPulseTemplate':
        subtemplates = [serializer.deserialize(st) for st in subtemplates]
        external_parameters = set.union(*(st.parameter_names for st in subtemplates))
        mul_template = MultiChannelPulseTemplate(subtemplates, external_parameters, identifier=identifier)
        return mul_template


class ChannelMappingException(Exception):
    def __init__(self, obj1, obj2, intersect_set):
        self.intersect_set = intersect_set
        self.obj1 = obj1
        self.obj2 = obj2
    def __str__(self):
        return 'Channels {chs} defined in {} and {}'.format(self.intersect_set, self.obj1, self.obj2)