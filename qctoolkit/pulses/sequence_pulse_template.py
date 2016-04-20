"""This module defines SequencePulseTemplate, a higher-order hierarchical pulse template that
combines several other PulseTemplate objects for sequential execution."""


from typing import Dict, List, Tuple, Set, Optional, Any, Iterable, Union

from qctoolkit.serialization import Serializer
from qctoolkit.expressions import Expression

from qctoolkit.pulses.pulse_template import PulseTemplate, MeasurementWindow, \
    DoubleParameterNameException
from qctoolkit.pulses.parameters import ParameterDeclaration, Parameter, \
    ParameterNotProvidedException, MappedParameter
from qctoolkit.pulses.sequencing import InstructionBlock, Sequencer
from qctoolkit.pulses.conditions import Condition

__all__ = ["SequencePulseTemplate",
           "MissingMappingException",
           "MissingParameterDeclarationException",
           "UnnecessaryMappingException"]


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
        self.__parameter_names = frozenset(external_parameters)
        # convert all mapping strings to expressions
        for i, (template, mappings) in enumerate(subtemplates):
            subtemplates[i] = (template, {k: Expression(v) for k, v in mappings.items()})

        for template, mapping_functions in subtemplates:
            # Consistency checks
            open_parameters = template.parameter_names
            mapped_parameters = set(mapping_functions.keys())
            missing_parameters = open_parameters - mapped_parameters
            for missing in missing_parameters:
                raise MissingMappingException(template, missing)
            unnecessary_parameters = mapped_parameters - open_parameters
            for unnecessary in unnecessary_parameters:
                raise UnnecessaryMappingException(template, unnecessary)

            for key, mapping_function in mapping_functions.items():
                mapping_function = mapping_functions[key]
                required_externals = set(mapping_function.variables())
                non_declared_externals = required_externals - self.__parameter_names
                if non_declared_externals:
                    raise MissingParameterDeclarationException(template,
                                                               non_declared_externals.pop())

        self.subtemplates = subtemplates  # type: List[Subtemplate]
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
