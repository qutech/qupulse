import logging
from typing import Dict, List, Tuple, Set, Callable, Optional, Any, Iterable, Union
from functools import partial
from inspect import getsource
import copy
from inspect import getargspec
"""RELATED THIRD PARTY IMPORTS"""
from py_expression_eval import Parser

"""LOCAL IMPORTS"""
from .PulseTemplate import PulseTemplate, MeasurementWindow, ParameterNotInPulseTemplateException
from .Parameter import ParameterDeclaration, Parameter, ParameterNotProvidedException
from .Sequencer import InstructionBlock, Sequencer
from .Serializer import Serializer


logger = logging.getLogger(__name__)

# type signatures used in this module
# a MappingFunction takes a dictionary with parameter declarations, keyed with strings and returns a float
# temporarily obsolete: MappingFunction = Callable[[Dict[str, ParameterDeclaration]], float]

# a subtemplate consists of a pulse template and mapping functions for its "internal" parameters
Subtemplate = Tuple[PulseTemplate, Dict[str, str]]

class SequencePulseTemplate(PulseTemplate):
    """A sequence of different PulseTemplates.
    
    SequencePulseTemplate allows to group smaller
    PulseTemplates (subtemplates) into on larger sequence,
    i.e., when instantiating a pulse from a SequencePulseTemplate
    all pulses instantiated from the subtemplates are queued for
    execution right after one another.
    SequencePulseTemplate allows to specify a mapping of
    parameter declarations from its subtemplates, enabling
    renaming and mathematical transformation of parameters.
    The default behavior is to exhibit the union of parameter
    declarations of all subtemplates. If two subpulses declare
    a parameter with the same name, it is mapped to both. If the
    declarations define different minimal and maximal values, the
    more restricitive is chosen, if possible. Otherwise, an error
    is thrown and an explicit mapping must be specified.
    ^outdated
    """

    def __init__(self, subtemplates: List[Subtemplate], external_parameters: List[str], identifier: Optional[str]=None) -> None:
        super().__init__(identifier)
        self.__parameter_names = frozenset(external_parameters)
        for template, mapping_functions in subtemplates:
            # Consistency checks
            open_parameters = template.parameter_names
            mapped_parameters = set(mapping_functions.keys())
            missing_parameters = open_parameters - mapped_parameters
            for m in missing_parameters:
                raise MissingMappingException(template, m)
            unnecessary_parameters = mapped_parameters - open_parameters
            for m in unnecessary_parameters:
                raise UnnecessaryMappingException(template, m)

            for mapping_function in mapping_functions.values():
                mapping_function = Parser().parse(mapping_function)
                required_externals = set(mapping_function.variables())
                non_declared_externals = required_externals - self.__parameter_names
                if non_declared_externals:
                    raise MissingParameterDeclarationException(template, non_declared_externals.pop())

        self.subtemplates = subtemplates  # type: List[Subtemplate]
        self.__is_interruptable = True

    @property
    def parameter_names(self) -> Set[str]:
        return self.__parameter_names

    @property
    def parameter_declarations(self) -> Set[ParameterDeclaration]:
        # TODO: min, max, default values not mapped (required?)
        return set([ParameterDeclaration(name) for name in self.__parameter_names])

    def get_measurement_windows(self, parameters: Dict[str, Parameter] = None) -> List[MeasurementWindow]:
        raise NotImplemented() # will be computed by Sequencer

    @property
    def is_interruptable(self) -> bool:
        return self.__is_interruptable
    
    @is_interruptable.setter
    def is_interruptable(self, new_value: bool):
        self.__is_interruptable = new_value

    def requires_stop(self, parameters: Dict[str, Parameter]) -> bool:
        if not self.subtemplates:
            return False

        # obtain first subtemplate
        (template, mapping_functions) = self.subtemplates[0]

        # collect all parameters required to compute the mappings for the first subtemplate
        external_parameters = set()
        for mapping_function in mapping_functions.values():
            mapping_function = Parser().parse(mapping_function)
            external_parameters = external_parameters | set([parameters[x] for x in mapping_function.variables()])

        # return True only if none of these requires a stop
        return any([p.requires_stop for p in external_parameters])

    def __map_parameter(self, mapping_function: str, parameters: Dict[str, Parameter]) -> Parameter:
        mapping_function = Parser().parse(mapping_function)
        external_parameters = mapping_function.variables()
        external_values = {name: float(parameters[name]) for name in external_parameters}
        return mapping_function.evaluate(external_values)

    def build_sequence(self, sequencer: Sequencer, parameters: Dict[str, Parameter], instruction_block: InstructionBlock) -> None:
        # detect missing or unnecessary parameters
        missing = self.parameter_names - set(parameters)
        for m in missing:
            raise ParameterNotProvidedException(m)

        # push subtemplates to sequencing stack with mapped parameters
        for template, mappings in reversed(self.subtemplates):
            inner_parameters = {name: self.__map_parameter(mapping_function, parameters) for (name, mapping_function) in mappings.items()}
            sequencer.push(template, inner_parameters, instruction_block)

    def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
        data = dict()
        data['external_parameters'] = sorted(list(self.parameter_names))
        data['is_interruptable'] = self.is_interruptable

        subtemplates = []
        for (subtemplate, mapping_functions) in self.subtemplates:
            subtemplate = serializer._serialize_subpulse(subtemplate)
            subtemplates.append(dict(template=subtemplate, mappings=dict(mapping_functions)))
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
             {k: m for k, m in d['mappings'].items()}) for d in subtemplates]

        template = SequencePulseTemplate(subtemplates, external_parameters, identifier=identifier)
        template.is_interruptable = is_interruptable
        return template


class MissingParameterDeclarationException(Exception):

    def __init__(self, template: PulseTemplate, missing_delcaration: str) -> None:
        super().__init__()
        self.template = template
        self.missing_declaration = missing_delcaration

    def __str__(self) -> str:
        return "A mapping for template {} requires a parameter '{}' which has not been declared as an external" \
               " parameter of the SequencePulseTemplate.".format(self.template, self.missing_declaration)


class MissingMappingException(Exception):

    def __init__(self, template, key) -> None:
        super().__init__()
        self.key = key
        self.template = template

    def __str__(self) -> str:
        return "The template {} needs a mapping function for parameter {}". format(self.template, self.key)


class UnnecessaryMappingException(Exception):

    def __init__(self, template, key):
        super().__init__()
        self.template = template
        self.key = key

    def __str__(self) -> str:
        return "Mapping function for parameter '{}', which template {} does not need".format(self.key, self.template)

class RuntimeMappingError(Exception):
    def __init__(self, template, subtemplate, outer_key, inner_key):
        self.template = template
        self.subtemplate = subtemplate
        self.outer_key = outer_key
        self.inner_key = inner_key

    def __str__(self):
        return ("An error occurred in the mapping function from {} to {}."
                " The mapping function for inner parameter '{}' requested"
                " outer parameter '{}', which was not provided.").format(self.template, self.subtemplate, self.inner_key, self.outer_key)
