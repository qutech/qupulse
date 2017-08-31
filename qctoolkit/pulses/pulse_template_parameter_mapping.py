"""This module defines PulseTemplateParameterMapping, a helper class for pulse templates that
offer mapping of parameters of subtemplates."""

from typing import Optional, Set, Dict, Union, List, Any, Tuple
import itertools
import numbers

from qctoolkit.utils.types import ChannelID
from qctoolkit.expressions import Expression
from qctoolkit.pulses.pulse_template import PulseTemplate
from qctoolkit.pulses.parameters import Parameter, MappedParameter, ParameterNotProvidedException, ParameterConstrainer
from qctoolkit.pulses.sequencing import Sequencer
from qctoolkit.pulses.instructions import InstructionBlock, Waveform
from qctoolkit.pulses.conditions import Condition
from qctoolkit.serialization import Serializer

__all__ = [
    "MappingPulseTemplate",
    "MissingMappingException",
    "MissingParameterDeclarationException",
    "UnnecessaryMappingException",
]


MappingTuple = Union[Tuple[PulseTemplate],
                     Tuple[PulseTemplate, Dict],
                     Tuple[PulseTemplate, Dict, Dict],
                     Tuple[PulseTemplate, Dict, Dict, Dict]]


class MappingPulseTemplate(PulseTemplate, ParameterConstrainer):
    """This class can be used to remap parameters, the names of measurement windows and the names of channels. Besides
    the standard constructor, there is a static member function from_tuple for convenience. The class also allows
    constraining parameters by deriving from ParameterConstrainer"""
    def __init__(self, template: PulseTemplate, *,
                 identifier: Optional[str]=None,
                 parameter_mapping: Optional[Dict[str, str]]=None,
                 measurement_mapping: Optional[Dict[str, str]] = None,
                 channel_mapping: Optional[Dict[ChannelID, ChannelID]] = None,
                 parameter_constraints: Optional[List[str]]=None,
                 allow_partial_parameter_mapping=False):
        """Standard constructor for the MappingPulseTemplate.

        Mappings that are not specified are defaulted to identity mappings. Channels and measurement names of the
        encapsulated template can be mapped partially by default. F.i. if channel_mapping only contains one of two
        channels the other channel name is mapped to itself.
        However, if a parameter mapping is specified and one or more parameters are not mapped a MissingMappingException
        is raised. To allow partial mappings and enable the same behaviour as for the channel and measurement name
        mapping allow_partial_parameter_mapping must be set to True.
        Furthermore parameter constrains can be specified.
        
        :param template: The encapsulated pulse template whose parameters, measurement names and channels are mapped
        :param parameter_mapping: if not none, mappings for all parameters must be specified
        :param measurement_mapping: mappings for other measurement names are inserted
        :param channel_mapping: mappings for other channels are auto inserted
        :param parameter_constraints:
        :param allow_partial_parameter_mapping:
        """
        PulseTemplate.__init__(self, identifier=identifier)
        ParameterConstrainer.__init__(self, parameter_constraints=parameter_constraints)

        if parameter_mapping is None:
            parameter_mapping = dict((par, par) for par in template.parameter_names)
        else:
            mapped_internal_parameters = set(parameter_mapping.keys())
            internal_parameters = template.parameter_names
            missing_parameter_mappings = internal_parameters - mapped_internal_parameters
            if mapped_internal_parameters - internal_parameters:
                raise UnnecessaryMappingException(template, mapped_internal_parameters - internal_parameters)
            elif missing_parameter_mappings:
                if allow_partial_parameter_mapping:
                    parameter_mapping.update({p: p for p in missing_parameter_mappings})
                else:
                    raise MissingMappingException(template, internal_parameters - mapped_internal_parameters)
        parameter_mapping = dict((k, Expression(v)) for k, v in parameter_mapping.items())

        measurement_mapping = dict() if measurement_mapping is None else measurement_mapping
        internal_names = template.measurement_names
        mapped_internal_names = set(measurement_mapping.keys())
        if mapped_internal_names - internal_names:
            raise UnnecessaryMappingException(template, mapped_internal_names - internal_names)
        missing_name_mappings = internal_names - mapped_internal_names
        measurement_mapping = dict(itertools.chain(((name, name) for name in missing_name_mappings),
                                                   measurement_mapping.items()))

        channel_mapping = dict() if channel_mapping is None else channel_mapping
        internal_channels = template.defined_channels
        mapped_internal_channels = set(channel_mapping.keys())
        if mapped_internal_channels - internal_channels:
            raise UnnecessaryMappingException(template,mapped_internal_channels - internal_channels)
        missing_channel_mappings = internal_channels - mapped_internal_channels
        channel_mapping = dict(itertools.chain(((name, name) for name in missing_channel_mappings),
                                               channel_mapping.items()))

        if isinstance(template, MappingPulseTemplate) and template.identifier is None:
            # avoid nested mappings
            parameter_mapping = {p: expr.evaluate_symbolic(parameter_mapping)
                                 for p, expr in template.parameter_mapping.items()}
            measurement_mapping = {k: measurement_mapping[v]
                                   for k, v in template.measurement_mapping.items()}
            channel_mapping = {k: channel_mapping[v]
                               for k, v in template.channel_mapping.items()}
            template = template.template

        self.__template = template
        self.__parameter_mapping = parameter_mapping
        self.__external_parameters = set(itertools.chain(*(expr.variables for expr in self.__parameter_mapping.values())))
        self.__external_parameters |= self.constrained_parameters
        self.__measurement_mapping = measurement_mapping
        self.__channel_mapping = channel_mapping

    @staticmethod
    def from_tuple(mapping_tuple: MappingTuple) -> 'MappingPulseTemplate':
        """Construct a MappingPulseTemplate from a tuple of mappings. The mappings are automatically assigned to the
        mapped elements based on their content.
        :param mapping_tuple: A tuple of mappings
        :return: Constructed MappingPulseTemplate
        """
        template, *mappings = mapping_tuple

        parameter_mapping = None
        measurement_mapping = None
        channel_mapping = None

        for mapping in mappings:
            if len(mapping) == 0:
                continue

            mapped = set(mapping.keys())
            if sum((mapped <= template.parameter_names,
                    mapped <= template.measurement_names,
                    mapped <= template.defined_channels)) > 1:
                raise AmbiguousMappingException(template, mapping)

            if mapped == template.parameter_names:
                if parameter_mapping:
                    raise MappingCollisionException(template, object_type='parameter',
                                                    mapped=template.parameter_names,
                                                    mappings=(parameter_mapping, mapping))
                parameter_mapping = mapping
            elif mapped <= template.measurement_names:
                if measurement_mapping:
                    raise MappingCollisionException(template, object_type='measurement',
                                                    mapped=template.measurement_names,
                                                    mappings=(measurement_mapping, mapping))
                measurement_mapping = mapping
            elif mapped <= template.defined_channels:
                if channel_mapping:
                    raise MappingCollisionException(template, object_type='channel',
                                                    mapped=template.defined_channels,
                                                    mappings=(channel_mapping, mapping))
                channel_mapping = mapping
            else:
                raise ValueError('Could not match mapping to mapped objects: {}'.format(mapping))
        return MappingPulseTemplate(template,
                                    parameter_mapping=parameter_mapping,
                                    measurement_mapping=measurement_mapping,
                                    channel_mapping=channel_mapping)

    @property
    def template(self) -> PulseTemplate:
        return self.__template

    @property
    def measurement_mapping(self) -> Dict[str, str]:
        return self.__measurement_mapping

    @property
    def parameter_mapping(self) -> Dict[str, Expression]:
        return self.__parameter_mapping

    @property
    def channel_mapping(self) -> Dict[ChannelID, ChannelID]:
        return self.__channel_mapping

    @property
    def parameter_names(self) -> Set[str]:
        return self.__external_parameters

    @property
    def measurement_names(self) -> Set[str]:
        return set(self.__measurement_mapping.values())

    @property
    def is_interruptable(self) -> bool:
        return self.template.is_interruptable  # pragma: no cover

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return {self.__channel_mapping[k] for k in self.template.defined_channels}

    @property
    def duration(self) -> Expression:
        return self.__template.duration.evaluate_symbolic(self.__parameter_mapping)

    def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
        parameter_mapping_dict = dict((key, str(expression)) for key, expression in self.__parameter_mapping.items())
        return dict(template=serializer.dictify(self.template),
                    parameter_mapping=parameter_mapping_dict,
                    measurement_mapping=self.__measurement_mapping,
                    channel_mapping=self.__channel_mapping)

    @staticmethod
    def deserialize(serializer: Serializer,
                    template: Union[str, Dict[str, Any]], **kwargs) -> 'MappingPulseTemplate':
        return MappingPulseTemplate(template=serializer.deserialize(template),
                                    **kwargs)

    def map_parameters(self,
                       parameters: Dict[str, Union[Parameter, numbers.Real]]) -> Dict[str, Parameter]:
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

        self.validate_parameter_constraints(parameters=parameters)
        if all(isinstance(parameter, Parameter) for parameter in parameters.values()):
            return {parameter: MappedParameter(mapping_function, {name: parameters[name]
                                                                  for name in mapping_function.variables})
                    for (parameter, mapping_function) in self.__parameter_mapping.items()}
        if all(isinstance(parameter, numbers.Real) for parameter in parameters.values()):
            return {parameter: mapping_function.evaluate_numeric(**parameters)
                    for parameter, mapping_function in self.__parameter_mapping.items()}
        raise TypeError('Values of parameter dict are neither all Parameter nor Real')

    def get_updated_measurement_mapping(self, measurement_mapping: Dict[str, str]) -> Dict[str, str]:
        return {k: measurement_mapping[v] for k, v in self.__measurement_mapping.items()}

    def get_updated_channel_mapping(self, channel_mapping: Dict[ChannelID, ChannelID]) -> Dict[ChannelID, ChannelID]:
        return {inner_ch: channel_mapping[outer_ch] for inner_ch, outer_ch in self.__channel_mapping.items()}

    def build_sequence(self,
                       sequencer: Sequencer,
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, Condition],
                       measurement_mapping: Dict[str, str],
                       channel_mapping: Dict[ChannelID, ChannelID],
                       instruction_block: InstructionBlock) -> None:
        self.template.build_sequence(sequencer,
                                     parameters=self.map_parameters(parameters),
                                     conditions=conditions,
                                     measurement_mapping=self.get_updated_measurement_mapping(measurement_mapping),
                                     channel_mapping=self.get_updated_channel_mapping(channel_mapping),
                                     instruction_block=instruction_block)

    def build_waveform(self,
                       parameters: Dict[str, numbers.Real],
                       measurement_mapping: Dict[str, str],
                       channel_mapping: Dict[ChannelID, ChannelID]) -> Waveform:
        """This gets called if the parent is atomic"""
        return self.template.build_waveform(
            parameters=self.map_parameters(parameters),
            measurement_mapping=self.get_updated_measurement_mapping(measurement_mapping),
            channel_mapping=self.get_updated_channel_mapping(channel_mapping))

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, Condition]) -> bool:
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


class AutoMappingMatchingException(Exception):
    """Indicates that the auto match of mappings to mapped objects by the keys failed"""

    def __init__(self, template: PulseTemplate):
        super().__init__()
        self.template = template


class AmbiguousMappingException(AutoMappingMatchingException):
    """Indicates that a mapping may apply to multiple objects"""

    def __init__(self, template: PulseTemplate, mapping: Dict):
        super().__init__(template)
        self.mapping = mapping

    def __str__(self) -> str:
        return "Could not match mapping uniquely to object type: {}\nParameters: {}\nChannels: {}\nMeasurements: {}"\
            .format(self.mapping, self.template.parameter_names, self.template.defined_channels,
                    self.template.measurement_names)


class MappingCollisionException(AutoMappingMatchingException):
    """Indicates that multiple mappings are fitting for the same parameter type"""
    def __init__(self, template: PulseTemplate, object_type: str, mapped: Set, mappings: Tuple[Dict, ...]):
        super().__init__(template)
        self.parameter_type = object_type
        self.mappings = mappings
        self.message = 'Got multiple candidates for the {type} mapping.\nMapped: {mapped}\nCandidates:\n'\
            .format(type=object_type, mapped=mapped)

    def __str__(self) -> str:
        return self.message + '\n'.join(str(mapping) for mapping in self.mappings)
