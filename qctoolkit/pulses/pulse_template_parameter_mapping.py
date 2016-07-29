"""This module defines PulseTemplateParameterMapping, a helper class for pulse templates that
offer mapping of parameters of subtemplates."""

from typing import Optional, Set, Dict, Union

from qctoolkit.expressions import Expression
from qctoolkit.pulses.pulse_template import PulseTemplate
from qctoolkit.pulses.parameters import Parameter, MappedParameter, ParameterNotProvidedException

__all__ = [
    "MissingMappingException",
    "MissingParameterDeclarationException",
    "UnnecessaryMappingException",
    "PulseTemplateParameterMapping"
]


class PulseTemplateParameterMapping:
    """A mapping of parameters of a set of pulse templates to a fixed set of so-called external
    parameters.

    A helper class used by templates that offer parameter mappings of subtemplates. Automatically
    performs sanity checks when a new pulse template parameter mapping is added:
    - Do all mapping expression only rely on external parameters as variables?
    and offers functionality to query whether all parameters declared by a pulse tempalte are
    mapped.

    See Also:
        - SequencePulseTemplate
        - MultiChannelPulseTemplate
    """

    def __init__(self,
                 external_parameters: Optional[Set[str]]=None) -> None:
        """Create a new PulseTemplateParameterMapping instance.

        Args:
            external_parameters (Set(str)): A set of names of external parameters (optional).
        """
        super().__init__()
        self.__map = dict()
        self.__external_parameters = set()
        self.set_external_parameters(external_parameters)

    def set_external_parameters(self, external_parameters: Optional[Set[str]]) -> None:
        """Sets the set of external parameters names.

        Args:
            external_parameters (Set(str)): A set of names of external parameters. Might be None,
                which results in no changes.
        """
        if external_parameters is not None:
            self.__external_parameters = set(external_parameters.copy())

    @property
    def external_parameters(self) -> Set[str]:
        """The (names of the) external parameters."""
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
        """Add a new mapping for a parameter of a pulse template.

        Args:
            template (PulseTemplate): The pulse template for which a parameter mapping will be
                added.
            parameter (str): The name of the parameter of the pulse template that will be mapped.
            mapping_expression (str or Expression): The mathematical expression that specifies the
                mapping from external parameters to the parameter of the pulse template.
        Raises:
            UnnecessaryMappingException, if parameter is not declared by template.
            MissingParameterDeclarationException, if mapping_expression requires a variable that
                is not a parameter in the external parameters of this PulseTemplateParameterMapping.
        """
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
        """Return all parameter mappings defined for a given pulse template.

        Args:
            template (PulseTemplate): The pulse template for which to query the mapping.
        Returns:
            A dictionary of the form template_parameter -> mapping_expression for all mappings
            given for template.
        """
        return self.__get_template_map(template).copy()

    def is_template_mapped(self, template: PulseTemplate) -> bool:
        """Query whether a complete parameter mapping is defined for a given pulse template.

        Args:
            template (PulseTemplate): The pulse template for which to query the existence of
                mappings.
        Returns:
            True, if all parameters of template are mapped to external parameters.
        """
        return len(self.get_remaining_mappings(template)) == 0

    def get_remaining_mappings(self, template: PulseTemplate) -> Set[str]:
        """Query all currently unmapped parameters of a given pulse template.

        Args:
            template (PulseTemplate): The pulse template for which to query the unmapped parameters.
        Returns:
            A set of parameter names for which no mappings are defined.
        """
        template_map = self.__get_template_map(template)
        return template.parameter_names - template_map.keys()

    def map_parameters(self,
                       template: PulseTemplate,
                       parameters: Dict[str, Parameter]) -> Dict[str, Parameter]:
        """Map parameter values according to the defined mappings for a given pulse template.

        Args:
            template (PulseTemplate): The pulse template for which to map the parameter values.
            parameters (Dict(str -> Parameter)): A mapping of parameter names to Parameter
                objects/values.
        Returns:
            A new dictionary which maps parameter names to parameter values which have been
            mapped according to the mappings defined for template.
        """
        missing = self.__external_parameters - set(parameters.keys())
        if missing:
            raise ParameterNotProvidedException(missing.pop())

        template_map = self.__get_template_map(template)
        inner_parameters = {
            parameter: MappedParameter(
                mapping_function,
                {name: parameters[name] for name in mapping_function.variables()}
            )
            for (parameter, mapping_function) in template_map.items()
        }
        return inner_parameters


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

    def __init__(self, template: PulseTemplate, key: str) -> None:
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
