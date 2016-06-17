"""This module defines SequencePulseTemplate, a higher-order hierarchical pulse template that
combines several other PulseTemplate objects for sequential execution."""


from typing import Dict, List, Tuple, Set, Optional, Any, Iterable, Union

from qctoolkit.serialization import Serializer
from qctoolkit.expressions import Expression

from qctoolkit.pulses.pulse_template import PulseTemplate, MeasurementWindow
from qctoolkit.pulses.parameters import ParameterDeclaration, Parameter, \
    ParameterNotProvidedException, MappedParameter
from qctoolkit.pulses.sequencing import InstructionBlock, Sequencer
from qctoolkit.pulses.conditions import Condition
from qctoolkit.serialization import Serializable

__all__ = ["SequencePulseTemplate",
           "MissingMappingException",
           "MissingParameterDeclarationException",
           "UnnecessaryMappingException"]


class PulseTemplateParameterMapping:

    def __init__(self,
                 external_parameters: Optional[Set[str]]=None) -> None:
        super().__init__()
        self.__map = dict()
        self.__external_parameters = {}
        self.set_external_parameters(external_parameters)

    def set_external_parameters(self, external_parameters: Optional[Set[str]]) -> None:
        if external_parameters:
            self.__external_parameters = set(external_parameters.copy())

    @property
    def external_parameters(self) -> Set[str]:
        return self.__external_parameters.copy()

    def __get_template_map(self, template: PulseTemplate) -> Dict[str, Expression]:
        # internal helper function
        if template not in self.__map:
            return dict()
        return self.__map[template]

    def add(self,
            template: PulseTemplate,
            parameter: str,
            mapping_expression: Union[str, Expression]):
        if parameter not in template.parameter_names:
            raise UnnecessaryMappingException(template, parameter)

        if isinstance(mapping_expression, str):
            mapping_expression = Expression(mapping_expression)
        required_externals = set(mapping_expression.variables())
        non_declared_externals = required_externals - self.__external_parameters
        if non_declared_externals:
            raise MissingParameterDeclarationException(template,
                                                       non_declared_externals.pop())

        template_map = self.__get_template_map(template)
        template_map[parameter] = mapping_expression
        self.__map[template] = template_map

    def get_template_map(self, template: PulseTemplate) -> Dict[str, Expression]:
        return self.__get_template_map(template).copy()

    def is_template_mapped(self, template: PulseTemplate) -> bool:
        return self.remaining_mappings(template)

    def remaining_mappings(self, template: PulseTemplate) -> Set[str]:
        template_map = self.__get_template_map(template)
        return template.parameter_names - template_map.keys()

# a subtemplate consists of a pulse template and mapping functions for its "internal" parameters
Subtemplate = Tuple[PulseTemplate, Dict[str, str]] # pylint: disable=invalid-name


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

    def __init__(self,
                 subtemplates: List[Subtemplate],
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
        """
        super().__init__(identifier)

        num_channels = 0
        if subtemplates:
            num_channels = subtemplates[0][0].num_channels

        self.__parameter_mapping = PulseTemplateParameterMapping(external_parameters)

        for template, mapping_functions in subtemplates:
            # Consistency checks
            if template.num_channels != num_channels:
                raise ValueError("Subtemplates have different number of channels!")

            for parameter, mapping_function in mapping_functions.items():
                self.__parameter_mapping.add(template, parameter, mapping_function)

            remaining = self.__parameter_mapping.remaining_mappings(template)
            if remaining:
                raise MissingMappingException(template,
                                              remaining.pop())

        self.subtemplates = [(template, self.__parameter_mapping.get_template_map(template)) for (template, _) in subtemplates]
        self.__parameter_names = self.__parameter_mapping.external_parameters
        self.__is_interruptable = True

    @property
    def parameter_names(self) -> Set[str]:
        return self.__parameter_names

    @property
    def parameter_declarations(self) -> Set[ParameterDeclaration]:
        # TODO: min, max, default values not mapped (required?)
        return set([ParameterDeclaration(name) for name in self.__parameter_names])

    def get_measurement_windows(self,
                                parameters: Dict[str, Parameter]=None
                                ) -> List[MeasurementWindow]:
        raise NotImplementedError() # will be computed by Sequencer

    @property
    def is_interruptable(self) -> bool:
        return self.__is_interruptable
    
    @is_interruptable.setter
    def is_interruptable(self, new_value: bool) -> None:
        self.__is_interruptable = new_value

    @property
    def num_channels(self) -> int:
        return self.subtemplates[0][0].num_channels

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, 'Condition']) -> bool:
        return False

    def build_sequence(self,
                       sequencer: Sequencer,
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, Condition],
                       instruction_block: InstructionBlock) -> None:
        # todo: currently ignores is_interruptable

        # detect missing or unnecessary parameters
        missing = list(self.parameter_names - set(parameters))
        if missing:
            raise ParameterNotProvidedException(missing[0])

        # push subtemplates to sequencing stack with mapped parameters
        for template, mappings in reversed(self.subtemplates):
            inner_parameters = {
                name: MappedParameter(
                    mapping_function,
                    {name: parameters[name] for name in mapping_function.variables()}
                )
                for (name, mapping_function) in mappings.items()
            }
            sequencer.push(template, inner_parameters, conditions, instruction_block)

    def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
        data = dict()
        data['external_parameters'] = sorted(list(self.parameter_names))
        data['is_interruptable'] = self.is_interruptable

        subtemplates = []
        for (subtemplate, mapping_functions) in self.subtemplates:
            mapping_functions_strings = \
                {k: serializer.dictify(m) for k, m in mapping_functions.items()}
            subtemplate = serializer.dictify(subtemplate)
            subtemplates.append(dict(template=subtemplate, mappings=mapping_functions_strings))
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
            [(serializer.deserialize(d['template']),
             {k: str(serializer.deserialize(m))
              for k, m in d['mappings'].items()})
             for d in subtemplates]

        template = SequencePulseTemplate(subtemplates, external_parameters, identifier=identifier)
        template.is_interruptable = is_interruptable
        return template


class MissingParameterDeclarationException(Exception):
    """Indicates that a parameter declaration mapping in a SequencePulseTemplate maps to an external
    parameter declaration that was not declared."""

    def __init__(self, template: PulseTemplate, missing_declaration: str) -> None:
        super().__init__()
        self.template = template
        self.missing_declaration = missing_declaration

    def __str__(self) -> str:
        return \
            "A mapping for template {} requires a parameter '{}' which has not been declared as" \
            " an external parameter of the SequencePulseTemplate.".format(
                self.template, self.missing_declaration
            )


class MissingMappingException(Exception):
    """Indicates that no mapping was specified for some parameter declaration of a
    SequencePulseTemplate's subtemplate."""

    def __init__(self, template, key) -> None:
        super().__init__()
        self.key = key
        self.template = template

    def __str__(self) -> str:
        return "The template {} needs a mapping function for parameter {}".\
            format(self.template, self.key)


class UnnecessaryMappingException(Exception):
    """Indicates that a mapping was provided that does not correspond to any of a
    SequencePulseTemplate's subtemplate's parameter declarations and is thus obsolete."""

    def __init__(self, template: PulseTemplate, key: str) -> None:
        super().__init__()
        self.template = template
        self.key = key

    def __str__(self) -> str:
        return "Mapping function for parameter '{}', which template {} does not need"\
            .format(self.key, self.template)
