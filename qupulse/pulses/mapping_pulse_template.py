from typing import Optional, Set, Dict, Union, List, Any, Tuple
import itertools
import numbers
import collections

from qupulse.utils.types import ChannelID, FrozenDict, FrozenMapping
from qupulse.expressions import Expression, ExpressionScalar
from qupulse.parameter_scope import Scope, MappedScope
from qupulse.pulses.pulse_template import PulseTemplate, MappingTuple
from qupulse.pulses.parameters import Parameter, MappedParameter, ParameterNotProvidedException, ParameterConstrainer
from qupulse._program.waveforms import Waveform
from qupulse._program._loop import Loop
from qupulse.serialization import Serializer, PulseRegistryType

__all__ = [
    "MappingPulseTemplate",
    "MissingMappingException",
    "UnnecessaryMappingException",
]


class MappingPulseTemplate(PulseTemplate, ParameterConstrainer):
    """This class can be used to remap parameters, the names of measurement windows and the names of channels. Besides
    the standard constructor, there is a static member function from_tuple for convenience. The class also allows
    constraining parameters by deriving from ParameterConstrainer"""

    ALLOW_PARTIAL_PARAMETER_MAPPING = True
    """Default value for allow_partial_parameter_mapping of the __init__ method."""

    def __init__(self, template: PulseTemplate, *,
                 identifier: Optional[str]=None,
                 parameter_mapping: Optional[Dict[str, str]]=None,
                 measurement_mapping: Optional[Dict[str, str]] = None,
                 channel_mapping: Optional[Dict[ChannelID, ChannelID]] = None,
                 parameter_constraints: Optional[List[str]]=None,
                 allow_partial_parameter_mapping: bool = None,
                 registry: PulseRegistryType=None) -> None:
        """Standard constructor for the MappingPulseTemplate.

        Mappings that are not specified are defaulted to identity mappings. Channels and measurement names of the
        encapsulated template can be mapped partially by default. F.i. if channel_mapping only contains one of two
        channels the other channel name is mapped to itself. Channels that are mapped to None are dropped.
        However, if a parameter mapping is specified and one or more parameters are not mapped a MissingMappingException
        is raised. To allow partial mappings and enable the same behaviour as for the channel and measurement name
        mapping allow_partial_parameter_mapping must be set to True.
        Furthermore parameter constrains can be specified.
        
        :param template: The encapsulated pulse template whose parameters, measurement names and channels are mapped
        :param parameter_mapping: if not none, mappings for all parameters must be specified
        :param measurement_mapping: mappings for other measurement names are inserted
        :param channel_mapping: mappings for other channels are auto inserted. Mapping to None drops the channel.
        :param parameter_constraints:
        :param allow_partial_parameter_mapping: If None the value of the class variable ALLOW_PARTIAL_PARAMETER_MAPPING
        """
        PulseTemplate.__init__(self, identifier=identifier)
        ParameterConstrainer.__init__(self, parameter_constraints=parameter_constraints)

        if allow_partial_parameter_mapping is None:
            allow_partial_parameter_mapping = self.ALLOW_PARTIAL_PARAMETER_MAPPING

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

        # we copy to modify in place
        channel_mapping = dict() if channel_mapping is None else channel_mapping.copy()
        internal_channels = template.defined_channels
        mapped_internal_channels = set(channel_mapping.keys())
        if mapped_internal_channels - internal_channels:
            raise UnnecessaryMappingException(template,mapped_internal_channels - internal_channels)

        # fill up implicit mappings (unchanged channels)
        missing_channel_mappings = internal_channels - mapped_internal_channels
        for name in missing_channel_mappings:
            channel_mapping[name] = name

        # None is an allowed overlapping target as it marks dropped channels
        overlapping_targets = {channel
                               for channel, n in collections.Counter(channel_mapping.values()).items()
                               if n > 1 and channel is not None}
        if overlapping_targets:
            raise ValueError('Cannot map multiple channels to the same target(s) %r' % overlapping_targets,
                             channel_mapping)

        if isinstance(template, MappingPulseTemplate) and template.identifier is None:
            # avoid nested mappings
            parameter_mapping = {p: Expression(expr.evaluate_symbolic(parameter_mapping))
                                 for p, expr in template.parameter_mapping.items()}
            measurement_mapping = {k: measurement_mapping[v]
                                   for k, v in template.measurement_mapping.items()}
            channel_mapping = {k: channel_mapping[v]
                               for k, v in template.channel_mapping.items()}
            template = template.template

        self.__template = template
        self.__parameter_mapping = FrozenDict(parameter_mapping)
        self.__external_parameters = set(itertools.chain(*(expr.variables for expr in self.__parameter_mapping.values())))
        self.__external_parameters |= self.constrained_parameters
        self.__measurement_mapping = measurement_mapping
        self.__channel_mapping = channel_mapping
        self._register(registry=registry)

    @classmethod
    def from_tuple(cls, mapping_tuple: MappingTuple) -> 'MappingPulseTemplate':
        """Construct a MappingPulseTemplate from a tuple of mappings. The mappings are automatically assigned to the
        mapped elements based on their content.
        :param mapping_tuple: A tuple of mappings
        :return: Constructed MappingPulseTemplate
        """
        template, *mappings = mapping_tuple

        if not mappings:
            return template

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

            if mapped <= template.parameter_names:
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
        return cls(template,
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
    def parameter_mapping(self) -> FrozenMapping[str, Expression]:
        return self.__parameter_mapping

    @property
    def channel_mapping(self) -> Dict[ChannelID, Optional[ChannelID]]:
        return self.__channel_mapping

    @property
    def parameter_names(self) -> Set[str]:
        return self.__external_parameters

    @property
    def measurement_names(self) -> Set[str]:
        return set(self.__measurement_mapping.values())

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return {self.__channel_mapping[k] for k in self.template.defined_channels} - {None}

    @property
    def duration(self) -> Expression:
        return self.__template.duration.evaluate_symbolic(
            {parameter_name: expression.underlying_expression
             for parameter_name, expression in self.__parameter_mapping.items()}
        )

    def get_serialization_data(self, serializer: Optional[Serializer]=None) -> Dict[str, Any]:
        data = super().get_serialization_data(serializer)

        if serializer: # compatibility to old serialization routines, deprecated
            parameter_mapping_dict = dict((key, str(expression)) for key, expression in self.__parameter_mapping.items())
            data = dict(template=serializer.dictify(self.template),
                        parameter_mapping=parameter_mapping_dict,
                        measurement_mapping=self.__measurement_mapping,
                        channel_mapping=self.__channel_mapping)

        else:
            data['template'] = self.template
            if self.__parameter_mapping:
                data['parameter_mapping'] = self.__parameter_mapping
            if self.__measurement_mapping:
                data['measurement_mapping'] = self.__measurement_mapping
            if self.__channel_mapping:
                data['channel_mapping'] = self.__channel_mapping

        if self.parameter_constraints:
            data['parameter_constraints'] = [str(c) for c in self.parameter_constraints]

        return data

    @classmethod
    def deserialize(cls,
                    serializer: Optional[Serializer]=None, # compatibility to old serialization routines, deprecated
                    **kwargs) -> 'MappingPulseTemplate':
        if serializer: # compatibility to old serialization routines, deprecated
            kwargs['template'] = serializer.deserialize(kwargs["template"])
        return cls(**kwargs, allow_partial_parameter_mapping=True)
        # return MappingPulseTemplate(template=serializer.deserialize(template),
        #                             **kwargs)

    def _validate_parameters(self, parameters: Dict[str, Union[Parameter, numbers.Real]], volatile: Set[str]):
        missing = set(self.__external_parameters) - set(parameters.keys())
        if missing:
            raise ParameterNotProvidedException(missing.pop())
        self.validate_parameter_constraints(parameters=parameters, volatile=volatile)

    def map_parameter_values(self, parameters: Dict[str, numbers.Real],
                             volatile: Set[str] = frozenset()) -> Dict[str, numbers.Real]:
        """Map parameter values according to the defined mappings.

        Args:
            parameters: Dictionary with numeric values
            volatile(Optional): Forwarded to `validate_parameter_constraints`
        Returns:
            A new dictionary with mapped numeric values.
        """
        self._validate_parameters(parameters=parameters, volatile=volatile)
        return {parameter: mapping_function.evaluate_numeric(**parameters)
                for parameter, mapping_function in self.__parameter_mapping.items()}

    def map_parameter_objects(self, parameters: Dict[str, Parameter],
                              volatile: Set[str] = frozenset()) -> Dict[str, Parameter]:
        """Map parameter objects (instances of Parameter class) according to the defined mappings.

        Args:
            parameters: Dictionary with parameter objects
            volatile(Optional): Forwarded to `validate_parameter_constraints`
        Returns:
            A new dictionary with mapped parameter objects
        """
        self._validate_parameters(parameters=parameters, volatile=volatile)
        return {parameter: MappedParameter(mapping_function,
                                           {name: parameters[name] for name in mapping_function.variables})
                for (parameter, mapping_function) in self.__parameter_mapping.items()}

    def map_scope(self, scope: Scope) -> MappedScope:
        return MappedScope(scope=scope, mapping=self.__parameter_mapping)

    def map_parameters(self,
                       parameters: Dict[str, Union[Parameter, numbers.Real]]) -> Dict[str,
                                                                                      Union[Parameter, numbers.Real]]:
        """Map parameter values according to the defined mappings.

        Args:
            parameters: A mapping of parameter names to parameter objects/values.
        Returns:
            A new dictionary which maps parameter names to parameter values which have been
            mapped according to the mappings defined for template.
        """
        if not parameters and self.__parameter_mapping:
            raise ValueError('Cannot infer type of return value (numeric or symbolic)')

        elif all(isinstance(parameter, numbers.Real) for parameter in parameters.values()):
            return self.map_parameter_values(parameters=parameters)

        elif all(isinstance(parameter, Parameter) for parameter in parameters.values()):
            return self.map_parameter_objects(parameters=parameters)

        else:
            raise TypeError('Values of parameter dict are neither all Parameter nor Real')

    def get_updated_measurement_mapping(self, measurement_mapping: Dict[str, str]) -> Dict[str, str]:
        return {k: measurement_mapping[v] for k, v in self.__measurement_mapping.items()}

    def get_updated_channel_mapping(self, channel_mapping: Dict[ChannelID,
                                                                Optional[ChannelID]]) -> Dict[ChannelID,
                                                                                              Optional[ChannelID]]:
        # do not look up the mapped outer channel if it is None (this marks a deleted channel)
        return {inner_ch: None if outer_ch is None else channel_mapping[outer_ch]
                for inner_ch, outer_ch in self.__channel_mapping.items()}

    def _internal_create_program(self, *,
                                 scope: Scope,
                                 measurement_mapping: Dict[str, Optional[str]],
                                 channel_mapping: Dict[ChannelID, Optional[ChannelID]],
                                 global_transformation: Optional['Transformation'],
                                 to_single_waveform: Set[Union[str, 'PulseTemplate']],
                                 parent_loop: Loop) -> None:
        self.validate_scope(scope)

        # parameters are validated in map_parameters() call, no need to do it here again explicitly
        self.template._create_program(scope=self.map_scope(scope),
                                      measurement_mapping=self.get_updated_measurement_mapping(measurement_mapping),
                                      channel_mapping=self.get_updated_channel_mapping(channel_mapping),
                                      global_transformation=global_transformation,
                                      to_single_waveform=to_single_waveform,
                                      parent_loop=parent_loop)

    def build_waveform(self,
                       parameters: Dict[str, numbers.Real],
                       channel_mapping: Dict[ChannelID, ChannelID]) -> Waveform:
        """This gets called if the parent is atomic"""
        return self.template.build_waveform(
            parameters=self.map_parameter_values(parameters),
            channel_mapping=self.get_updated_channel_mapping(channel_mapping))

    def get_measurement_windows(self,
                                parameters: Dict[str, numbers.Real],
                                measurement_mapping: Dict[str, Optional[str]]) -> List:
        return self.template.get_measurement_windows(
            parameters=self.map_parameter_values(parameters=parameters),
            measurement_mapping=self.get_updated_measurement_mapping(measurement_mapping=measurement_mapping)
        )

    def _apply_mapping_to_inner_channel_dict(self, to_map: Dict[ChannelID, ExpressionScalar]) -> Dict[ChannelID, ExpressionScalar]:
        parameter_mapping = {parameter_name: expression.underlying_expression
                             for parameter_name, expression in self.__parameter_mapping.items()}
        return {
            self.__channel_mapping.get(ch, ch): ExpressionScalar(ch_expr.sympified_expression.subs(parameter_mapping,
                                                                                                   simultaneous=True))
            for ch, ch_expr in to_map.items()
            if self.__channel_mapping.get(ch, ch) is not None
        }

    @property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        return self._apply_mapping_to_inner_channel_dict(self.__template.integral)

    def _as_expression(self) -> Dict[ChannelID, ExpressionScalar]:
        return self._apply_mapping_to_inner_channel_dict(self.__template._as_expression())

    @property
    def initial_values(self) -> Dict[ChannelID, ExpressionScalar]:
        return self._apply_mapping_to_inner_channel_dict(self.__template.initial_values)

    @property
    def final_values(self) -> Dict[ChannelID, ExpressionScalar]:
        return self._apply_mapping_to_inner_channel_dict(self.__template.final_values)


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
