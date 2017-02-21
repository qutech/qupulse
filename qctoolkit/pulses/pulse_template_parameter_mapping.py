"""This module defines PulseTemplateParameterMapping, a helper class for pulse templates that
offer mapping of parameters of subtemplates."""

from typing import Optional, Set, Dict, Union, Iterable, List, Any
import itertools

from qctoolkit import ChannelID
from qctoolkit.expressions import Expression
from qctoolkit.pulses.pulse_template import PulseTemplate, PossiblyAtomicPulseTemplate
from qctoolkit.pulses.parameters import Parameter, ParameterDeclaration, MappedParameter, ParameterNotProvidedException


__all__ = [
    "MappingTemplate",
    "MissingMappingException",
    "MissingParameterDeclarationException",
    "UnnecessaryMappingException",
]


class MappingTemplate(PossiblyAtomicPulseTemplate):
    def __init__(self, template: PulseTemplate,
                 parameter_mapping: Dict[str, str],
                 measurement_mapping: Dict[str, str] = dict(),
                 channel_mapping: Dict[ChannelID, ChannelID] = dict()):
        super().__init__(None)

        mapped_internal_parameters = set(parameter_mapping.keys())
        internal_parameters = template.parameter_names
        if mapped_internal_parameters - internal_parameters:
            raise UnnecessaryMappingException(template, mapped_internal_parameters - internal_parameters)
        elif internal_parameters - mapped_internal_parameters:
            raise MissingMappingException(template, internal_parameters - mapped_internal_parameters)
        parameter_mapping = dict((k, Expression(v)) for k, v in parameter_mapping.items())

        internal_names = template.measurement_names
        mapped_internal_names = set(measurement_mapping.keys())
        if mapped_internal_names - internal_names:
            raise UnnecessaryMappingException(template, mapped_internal_names - internal_names)
        missing_name_mappings = internal_names - mapped_internal_names

        internal_channels = template.defined_channels
        mapped_internal_channels = set(channel_mapping.keys())
        if mapped_internal_channels - internal_channels:
            raise UnnecessaryMappingException(template,mapped_internal_channels - internal_channels)
        missing_channel_mappings = internal_channels - mapped_internal_channels

        self.__template = template
        self.__parameter_mapping = parameter_mapping
        self.__external_parameters = set(itertools.chain(*(expr.variables() for expr in self.__parameter_mapping.values())))
        self.__measurement_mapping = dict(((name,name) for name in missing_name_mappings), **measurement_mapping)
        self.__channel_mapping = dict(((name,name) for name in missing_channel_mappings), **channel_mapping)

    @property
    def template(self) -> PulseTemplate:
        return self.__template

    @property
    def measurement_mapping(self) -> Dict[str, str]:
        return self.__measurement_mapping

    @property
    def parameter_names(self) -> Set[str]:
        return self.__external_parameters

    @property
    def parameter_declarations(self) -> Set[ParameterDeclaration]:
        # TODO: min, max, default values not mapped (required?)
        return {ParameterDeclaration(name) for name in self.parameter_names}

    @property
    def measurement_names(self) -> Set[str]:
        return set(self.__measurement_mapping.values())

    @property
    def is_interruptable(self) -> bool:
        return self.template.is_interruptable

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return {self.__channel_mapping[k] for k in self.template.defined_channels}

    @property
    def atomicity(self) -> bool:
        return self.__template.atomicity

    @atomicity.setter
    def atomicity(self, val) -> None:
        self.__template.atomicity = val

    def get_serialization_data(self, serializer: 'Serializer') -> Dict[str, Any]:
        parameter_mapping_dict = dict((key, str(expression)) for key, expression in self.__parameter_mapping.items())
        return dict(template=serializer.dictify(self.template),
                    parameter_mapping=parameter_mapping_dict,
                    measurement_mapping=self.__measurement_mapping,
                    channel_mapping=self.__channel_mapping)

    @staticmethod
    def deserialize(serializer: 'Serializer',
                    template: Union[str, Dict[str, Any]],
                    identifier: Optional[str]=None, **kwargs) -> 'MappingTemplate':
        return MappingTemplate(template=serializer.deserialize(template), **kwargs)

    def map_parameters(self,
                       parameters: Dict[str, Parameter]) -> Dict[str, Parameter]:
        """Map parameter values according to the defined mappings.

        Args:
            parameters (Dict(str -> Parameter)): A mapping of parameter names to Parameter
                objects/values.
        Returns:
            A new dictionary which maps parameter names to parameter values which have been
            mapped according to the mappings defined for template.
        """
        missing = set(self.__external_parameters) - set(parameters.keys())
        if missing:
            raise ParameterNotProvidedException(missing.pop())

        inner_parameters = {
            parameter: MappedParameter(
                mapping_function,
                {name: parameters[name] for name in mapping_function.variables()}
            )
            for (parameter, mapping_function) in self.__parameter_mapping.items()
        }
        return inner_parameters

    def get_updated_measurement_mapping(self, measurement_mapping: Dict[str, str]) -> Dict[str, str]:
        return {k: measurement_mapping[v] for k, v in self.__measurement_mapping.items()}

    def get_updated_channel_mapping(self, channel_mapping: Dict[ChannelID, ChannelID]) -> Dict[ChannelID, ChannelID]:
        return {inner_ch: channel_mapping[outer_ch] for inner_ch, outer_ch in self.__channel_mapping.items()}

    def build_sequence(self,
                       sequencer: "Sequencer",
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, 'Condition'],
                       measurement_mapping: Dict[str, str],
                       channel_mapping: Dict[ChannelID, ChannelID],
                       instruction_block: 'InstructionBlock') -> None:
        self.template.build_sequence(sequencer,
                                     parameters=self.map_parameters(parameters),
                                     conditions=conditions,
                                     measurement_mapping=self.get_updated_measurement_mapping(measurement_mapping),
                                     channel_mapping=self.get_updated_channel_mapping(channel_mapping),
                                     instruction_block=instruction_block)

    def build_waveform(self,
                       parameters: Dict[str, Parameter],
                       measurement_mapping: Dict[str, str],
                       channel_mapping: Dict[ChannelID, ChannelID]) -> 'Waveform':
        """This gets called if the parent is atomic"""
        return self.template.build_waveform(
            parameters=self.map_parameters(parameters),
            measurement_mapping=self.get_updated_measurement_mapping(measurement_mapping),
            channel_mapping=self.get_updated_channel_mapping(channel_mapping))

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, 'Condition']) -> bool:
        return self.template.requires_stop(
            self.map_parameters(parameters),
            conditions
        )


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

    def __init__(self, template: PulseTemplate, key: Union[str,Set[str]]) -> None:
        super().__init__()
        self.key = key
        self.template = template

    def __str__(self) -> str:
        return "The template {} needs a mapping function for parameter(s) {}".\
            format(self.template, self.key)


class UnnecessaryMappingException(Exception):
    """Indicates that a mapping was provided that does not correspond to any of a
    SequencePulseTemplate's subtemplate's parameter declarations and is thus obsolete."""

    def __init__(self, template: PulseTemplate, key: Union[str, Set[str]]) -> None:
        super().__init__()
        self.template = template
        self.key = key

    def __str__(self) -> str:
        return "Mapping function for parameter(s) '{}', which template {} does not need"\
            .format(self.key, self.template)
