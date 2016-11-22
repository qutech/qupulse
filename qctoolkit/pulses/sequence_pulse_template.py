"""This module defines SequencePulseTemplate, a higher-order hierarchical pulse template that
combines several other PulseTemplate objects for sequential execution."""


from typing import Dict, List, Tuple, Set, Optional, Any, Iterable, Union

from qctoolkit.serialization import Serializer

from qctoolkit.pulses.pulse_template import PulseTemplate, MeasurementWindow, \
    DoubleParameterNameException, SubTemplate
from qctoolkit.pulses.parameters import ParameterDeclaration, Parameter, \
    ParameterNotProvidedException
from qctoolkit.pulses.sequencing import InstructionBlock, Sequencer
from qctoolkit.pulses.conditions import Condition
from qctoolkit.pulses.pulse_template_parameter_mapping import PulseTemplateParameterMapping, \
    MissingMappingException, get_measurement_name_mappings

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
                 subtemplates: Iterable[Union[SimpleSubTemplate,SubTemplate]],
                 external_parameters: List[str], # pylint: disable=invalid-sequence-index
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

        num_channels = 0
        if subtemplates:
            num_channels = subtemplates[0][0].num_channels

        subtemplates = [ st if isinstance(st,SubTemplate) else SubTemplate(*st) for st in subtemplates  ]

        self.__parameter_mapping = PulseTemplateParameterMapping(external_parameters)

        for template, parameter_mapping, _, channel_mapping in subtemplates:
            if channel_mapping:
                raise ValueError('Channel mapping not allowed (yet) in SequencePulseTemplate')

            # Consistency checks
            if template.num_channels != num_channels:
                raise ValueError("Subtemplates have different number of channels!")

            for parameter, parameter_mapping_function in parameter_mapping.items():
                self.__parameter_mapping.add(template, parameter, parameter_mapping_function)

            remaining = self.__parameter_mapping.get_remaining_mappings(template)
            if remaining:
                raise MissingMappingException(template,
                                              remaining.pop())

        self.__measurement_window_mappings = get_measurement_name_mappings(subtemplates)
        self.__subtemplates = [st.template for st in subtemplates]
        self.__is_interruptable = True

    @property
    def parameter_names(self) -> Set[str]:
        return self.__parameter_mapping.external_parameters

    @property
    def parameter_declarations(self) -> Set[ParameterDeclaration]:
        # TODO: min, max, default values not mapped (required?)
        return {ParameterDeclaration(name) for name in self.parameter_names}

    @property
    def subtemplates(self) -> List[SubTemplate]:
        return [SubTemplate(template,
                            self.__parameter_mapping.get_template_map(template),
                            measurement_mapping=name_mapping)
                for template, name_mapping in zip(self.__subtemplates, self.__measurement_window_mappings)]

    @property
    def is_interruptable(self) -> bool:
        return self.__is_interruptable
    
    @is_interruptable.setter
    def is_interruptable(self, new_value: bool) -> None:
        self.__is_interruptable = new_value

    @property
    def num_channels(self) -> int:
        return self.__subtemplates[0].num_channels

    @property
    def measurement_names(self) -> Set[str]:
        return set.union(*(set(measurement_mapping.values())
                           for measurement_mapping in self.__measurement_window_mappings))

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, 'Condition']) -> bool:
        return False

    def build_sequence(self,
                       sequencer: Sequencer,
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, Condition],
                       measurement_mapping : Dict[str, str],
                       instruction_block: InstructionBlock) -> None:
        # todo: currently ignores is_interruptable

        # detect missing or unnecessary parameters
        missing = self.parameter_names - parameters.keys()
        if missing:
            raise ParameterNotProvidedException(missing.pop())

        def concatenate_dicts(d1, d2):
            return dict(((key, d1[value]) for key, value in d2.items()))

        # push subtemplates to sequencing stack with mapped parameters
        for template, local_window_mapping in zip(reversed(self.__subtemplates),reversed(self.__measurement_window_mappings)):
            inner_parameters = self.__parameter_mapping.map_parameters(template, parameters)
            inner_names = concatenate_dicts(measurement_mapping, local_window_mapping)
            sequencer.push(template, inner_parameters, conditions,
                           inner_names, instruction_block)

    def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
        data = dict()
        data['external_parameters'] = sorted(list(self.parameter_names))
        data['is_interruptable'] = self.is_interruptable

        subtemplates = []
        for subtemplate,window_name_mappings in zip(self.__subtemplates,self.__measurement_window_mappings):
            mapping_functions = self.__parameter_mapping.get_template_map(subtemplate)
            mapping_functions_strings = \
                {k: serializer.dictify(m) for k, m in mapping_functions.items()}
            subtemplate = serializer.dictify(subtemplate)
            subtemplates.append(dict(template=subtemplate, parameter_mappings=mapping_functions_strings,
                                     measurement_mappings=window_name_mappings))
        data['subtemplates'] = subtemplates

        data['type'] = serializer.get_type_identifier(self)
        return data

    @staticmethod
    def deserialize(serializer: Serializer,
                    is_interruptable: bool,
                    subtemplates: Iterable[Dict[str, Union[str, Dict[str, Any]]]],
                    external_parameters: Iterable[str],
                    identifier: Optional[str]=None) -> 'SequencePulseTemplate':
        subtemplates = \
            [SubTemplate(
                serializer.deserialize(d['template']),
                {k: str(serializer.deserialize(m)) for k, m in d['parameter_mappings'].items()},
                measurement_mapping=d['measurement_mappings']
            ) for d in subtemplates]

        template = SequencePulseTemplate(subtemplates, external_parameters, identifier=identifier)
        template.is_interruptable = is_interruptable
        return template

    def __matmult__(self, other) -> 'SequencePulseTemplate':
        """Like in the general PulseTemplate implementation this method enables using the
        @-operator for concatenating pulses. We need an overloaded method for SequencePulseTemplate
         to avoid creating unnecessarily nested pulse structures."""
        if not type(other) == SequencePulseTemplate:
            return SequencePulseTemplate.__matmult__(self, other)
        else:
            # this section is copy-pasted from the PulseTemplate implementation
            double_parameters = self.parameter_names & other.parameter_names  # intersection
            if double_parameters:
                raise DoubleParameterNameException(self, other, double_parameters)
            else:
                # this branch differs from what happens in PulseTemplate
                subtemplates = self.subtemplates + other.subtemplates
                # the check for conflicting external parameters has already been carried out
                external_parameters = self.parameter_names | other.parameter_names  # union
                return SequencePulseTemplate(subtemplates, external_parameters)  # no identifier

