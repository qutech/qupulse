"""This module defines SequencePulseTemplate, a higher-order hierarchical pulse template that
combines several other PulseTemplate objects for sequential execution."""


from typing import Dict, List, Tuple, Set, Optional, Any, Iterable, Union

from qctoolkit.serialization import Serializer

from qctoolkit.pulses.pulse_template import PulseTemplate, DoubleParameterNameException
from qctoolkit.pulses.parameters import ParameterDeclaration, Parameter
from qctoolkit.pulses.sequencing import InstructionBlock, Sequencer
from qctoolkit.pulses.conditions import Condition
from qctoolkit.pulses.pulse_template_parameter_mapping import \
    MissingMappingException, MappingTemplate, ChannelID, MissingParameterDeclarationException

__all__ = ["SequencePulseTemplate"]


class SequencePulseTemplate(PulseTemplate):
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

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, 'Condition']) -> bool:
        """Returns the stop requirement of the first subtemplate. If a later subtemplate requires a stop the
        SequencePulseTemplate can be partially sequenced."""
        return self.__subtemplates[0].requires_stop(parameters,conditions) if self.__subtemplates else False

    def build_sequence(self,
                       sequencer: Sequencer,
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, Condition],
                       measurement_mapping : Dict[str, str],
                       channel_mapping: Dict['ChannelID', 'ChannelID'],
                       instruction_block: InstructionBlock) -> None:
        # todo: currently ignores is_interruptable

        for subtemplate in reversed(self.subtemplates):
            sequencer.push(subtemplate, parameters, conditions, measurement_mapping, channel_mapping, instruction_block)

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
