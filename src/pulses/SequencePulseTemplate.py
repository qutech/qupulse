import logging
from typing import Dict, List, Tuple, Set, Callable, Optional, Any, Iterable, Union
from functools import partial
from inspect import getsource
import copy
"""RELATED THIRD PARTY IMPORTS"""

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
    """

    def __init__(self, subtemplates: List[Subtemplate], external_parameters: List[str], identifier: Optional[str]=None) -> None:
        super().__init__(identifier)
        self.__parameter_names = frozenset(external_parameters)
        for template, mapfuns in subtemplates:
            # Consistency checks
            open_parameters = template.parameter_names
            mapped_parameters = set(mapfuns.keys())
            missing_parameters = open_parameters - mapped_parameters
            for m in missing_parameters:
                raise MissingMappingException(template, m)
            unnecessary_parameters = mapped_parameters - open_parameters
            for m in unnecessary_parameters:
                raise UnnecessaryMappingException(template, m)

        self.subtemplates = subtemplates  # type: List[Subtemplate]
        self.__is_interruptable = True

    @property
    def parameter_names(self) -> Set[str]:
        return self.__parameter_names

    @property
    def parameter_declarations(self) -> Set[ParameterDeclaration]:
        parameter_declarations = set()
        for subtemplate in self.subtemplates:
            for declaration in subtemplate.parameter_declarations:
                if declaration.name in self.mapping.get_targets():
                    #Rename the parameter according to the mapping. The target name becomes the source name.
                    declaration = ParameterDeclaration(
                                                       self.mapping.get_targets()[declaration.name], 
                                                       min=declaration.min_value, 
                                                       max=declaration.max_value, 
                                                       default=declaration.default_value
                                                       ) 
                    # TODO: min, max and default values might have to be mapped to? especially in the case that min, max are ParameterDeclarations as well
                parameter_declarations.add(declaration)
        return parameter_declarations

    @property
    def measurement_windows(self, parameters: Dict[str, Parameter] = None) -> List[MeasurementWindow]:
        measurement_windows = []
        for subtemplate, _ in self.subtemplates:
            measurement_windows = measurement_windows + subtemplate.measurement_windows(parameters)
        return measurement_windows

    def get_measurement_windows(self, parameters: Dict[str, Parameter] = None):
        return self.measurement_windows

    @property
    def is_interruptable(self) -> bool:
        return self.__is_interruptable
    
    @is_interruptable.setter
    def is_interruptable(self, new_value: bool):
        self.__is_interruptable = new_value

    def requires_stop(self, parameters: Dict[str, Parameter]) -> bool:
       """IMPLEMENT ME!"""

    def build_sequence(self, sequencer: "Sequencer", parameters: Dict[str, Parameter], instruction_block: InstructionBlock) -> None:
        # detect missing or unnecessary parameters
        missing = self.parameter_names - set(parameters)
        for m in missing:
            raise ParameterNotProvidedException(m)

        # push subtemplates to sequencing stack with mapped parameters
        for template, mappings in reversed(self.subtemplates):
            # explicit looping for better error handling
            inner_parameters = {}
            for name in template.parameter_names:
                try:
                    value = mappings[name](parameters)
                except KeyError as e:
                    raise RuntimeMappingError(self, template, e.args[0], name) from e
                inner_parameters[name] = value
            sequencer.push(template, inner_parameters, instruction_block)

    def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
        data = dict()
        data['external_parameters'] = list(self.parameter_names)
        data['is_interruptable'] = self.is_interruptable

        subtemplates = []
        for (subtemplate, mapping_functions) in self.subtemplates:
            subtemplate = serializer._serialize_subpulse(subtemplate)
            #subtemplates.append((subtemplate, copy.deepcopy(mapping_functions)))
            subtemplates.append((subtemplate, {k: '<lambda>' for (k, m) in mapping_functions.items()}))
        data['subtemplates'] = subtemplates

        data['type'] = serializer.get_type_identifier(self)
        return data

    @staticmethod
    def deserialize(serializer: Serializer,
                    is_interruptable: bool,
                    subtemplates: Iterable[Tuple[Union[str, Dict[str, Any]], Dict[str, str]]],
                    external_parameters: Iterable[str],
                    identifier: Optional[str]=None) -> 'SequencePulseTemplate':
        subtemplates = [(serializer.deserialize(subtemplate), {k: lambda x: x for k, m in parameter_mappings.items()}) for subtemplate, parameter_mappings in subtemplates]

        template = SequencePulseTemplate(subtemplates, external_parameters, identifier=identifier)
        template.is_interruptable = is_interruptable
        return template


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
