
from typing import Any, Dict, List, Set, Optional, Union, Mapping, FrozenSet, cast, Callable
from numbers import Real
import warnings

import sympy
import cached_property

from qupulse.expressions import ExpressionScalar, ExpressionLike
from qupulse.serialization import Serializer, PulseRegistryType
from qupulse.parameter_scope import Scope

from qupulse.utils.types import ChannelID
from qupulse.pulses.parameters import Parameter
from qupulse.pulses.pulse_template import AtomicPulseTemplate, PulseTemplate
from qupulse._program.waveforms import Waveform, ArithmeticWaveform, TransformingWaveform
from qupulse._program.transformation import Transformation, ScalingTransformation, OffsetTransformation,\
    IdentityTransformation


class ArithmeticAtomicPulseTemplate(AtomicPulseTemplate):
    def __init__(self,
                 lhs: AtomicPulseTemplate,
                 arithmetic_operator: str,
                 rhs: AtomicPulseTemplate,
                 *,
                 silent_atomic: bool = False,
                 measurements: List = None,
                 identifier: str = None,
                 registry: PulseRegistryType = None):
        """Apply an operation (+ or -) channel wise to two atomic pulse templates. Channels only present in one pulse
        template have the operations neutral element on the other. The operations are defined in
        `ArithmeticWaveform.operator_map`.

        Non-atomic pulse templates are implicitly interpreted as atomic.

        Args:
            lhs: Left hand side operand
            arithmetic_operator: String representation of the operator
            rhs: Right hand side operand
            measurements: See AtomicPulseTemplate
            identifier: See AtomicPulseTemplate
            registry: See qupulse.serialization.PulseRegistry
        """
        super().__init__(identifier=identifier, measurements=measurements)

        if arithmetic_operator not in ArithmeticWaveform.operator_map:
            raise ValueError('Unknown operator. allowed: %r' % set(ArithmeticWaveform.operator_map.keys()))

        if lhs.duration != rhs.duration:
            warnings.warn("The operands have unequal expressions for their duration. "
                          "If they evaluate to different values on instantiation this will result in an error. "
                          "(%r != %r) for ALL inputs "
                          "(it may be unequal only for fringe cases)" % (lhs.duration, rhs.duration),
                          category=UnequalDurationWarningInArithmeticPT)

        if not silent_atomic and (not isinstance(lhs, AtomicPulseTemplate) or not isinstance(rhs, AtomicPulseTemplate)):
            warnings.warn("ArithmeticAtomicPulseTemplate treats all operands as if they are atomic. "
                          "You can silence this warning by passing `silent_atomic=True` or by ignoring this category.",
                          category=ImplicitAtomicityInArithmeticPT)

        self._lhs = lhs
        self._rhs = rhs
        self._arithmetic_operator = arithmetic_operator

        self._register(registry=registry)

    @property
    def lhs(self):
        return self._lhs

    @property
    def rhs(self):
        return self._rhs

    @property
    def arithmetic_operator(self) -> str:
        return self._arithmetic_operator

    @property
    def defined_channels(self):
        return self.lhs.defined_channels | self.rhs.defined_channels

    @property
    def parameter_names(self):
        return self.lhs.parameter_names | self.rhs.parameter_names

    @property
    def measurement_names(self):
        return super().measurement_names.union(self.lhs.measurement_names, self.rhs.measurement_names)

    @property
    def duration(self) -> ExpressionScalar:
        """Duration of the lhs operand if it is larger zero. Else duration of the rhs."""
        return ExpressionScalar(sympy.Max(self.lhs.duration, self.rhs.duration))

    @property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        lhs = self.lhs.integral
        rhs = self.rhs.integral

        result = lhs.copy()

        for channel, rhs_value in rhs.items():
            if channel in result:
                result[channel] = ArithmeticWaveform.operator_map[self._arithmetic_operator](result[channel], rhs_value)
            else:
                result[channel] = ArithmeticWaveform.rhs_only_map[self._arithmetic_operator](rhs_value)
        return result

    def build_waveform(self,
                       parameters: Dict[str, Real],
                       channel_mapping: Dict[ChannelID, Optional[ChannelID]]) -> Optional[Waveform]:
        lhs = self.lhs.build_waveform(parameters=parameters, channel_mapping=channel_mapping)
        rhs = self.rhs.build_waveform(parameters=parameters, channel_mapping=channel_mapping)

        if rhs is None:
            return lhs
        if lhs is None:
            return ArithmeticWaveform.rhs_only_map[self.arithmetic_operator](rhs)
        else:
            return ArithmeticWaveform(lhs, self.arithmetic_operator, rhs)

    def get_measurement_windows(self,
                                parameters: Dict[str, Real],
                                measurement_mapping: Dict[str, Optional[str]]) -> List['MeasurementWindow']:
        measurements = super().get_measurement_windows(parameters=parameters,
                                                       measurement_mapping=measurement_mapping)
        measurements.extend(self.lhs.get_measurement_windows(parameters=parameters,
                                                             measurement_mapping=measurement_mapping))

        measurements.extend(self.rhs.get_measurement_windows(parameters=parameters,
                                                             measurement_mapping=measurement_mapping))
        return measurements

    def get_serialization_data(self, serializer: Optional[Serializer]=None) -> Dict[str, Any]:
        data = super().get_serialization_data(serializer)
        data['rhs'] = self.rhs
        data['lhs'] = self.lhs
        data['arithmetic_operator'] = self.arithmetic_operator

        if serializer:
            raise NotImplementedError('Compatibility to old serialization routines not implemented for new type')

        if self.measurement_declarations:
            data['measurements'] = self.measurement_declarations

        return data

    def __repr__(self):
        if any(v for k, v in super().get_serialization_data().items() if k != '#type'):
            return super().__repr__()
        else:
            return '(%r %r %r)' % (self.lhs, self.arithmetic_operator, self.rhs)

    @classmethod
    def deserialize(cls, serializer: Optional[Serializer]=None, **kwargs) -> 'ArithmeticAtomicPulseTemplate':
        if serializer:
            raise NotImplementedError('Compatibility to old serialization routines not implemented for new type')

        return cls(**kwargs)


class ArithmeticPulseTemplate(PulseTemplate):
    """"""
    def __init__(self,
                 lhs: Union[PulseTemplate, ExpressionLike, Mapping[ChannelID, ExpressionLike]],
                 arithmetic_operator: str,
                 rhs: Union[PulseTemplate, ExpressionLike, Mapping[ChannelID, ExpressionLike]],
                 *,
                 identifier: Optional[str] = None):
        """

        Args:
            lhs: Left hand side operand
            arithmetic_operator: String representation of the operator
            rhs: Right hand side operand
            identifier:

        Raises:
             TypeError if both or none of the  operands are pulse templates
        """
        PulseTemplate.__init__(self, identifier=identifier)

        if not isinstance(lhs, PulseTemplate) and not isinstance(rhs, PulseTemplate):
            raise TypeError('At least one of the operands needs to be a pulse template.')

        elif not isinstance(lhs, PulseTemplate) and isinstance(rhs, PulseTemplate):
            #  +, - and * with (scalar, PT)
            if arithmetic_operator not in ('+', '-', '*'):
                raise ValueError('Operands (scalar, PulseTemplate) require an operator from {+, -, *}')
            scalar = lhs = self._parse_operand(lhs, rhs.defined_channels)
            pulse_template = rhs

        elif isinstance(lhs, PulseTemplate) and not isinstance(rhs, PulseTemplate):
            #  +, -, *, / and // with (PT, scalar)
            if arithmetic_operator not in ('+', '-', '*', '/'):
                raise ValueError('Operands (PulseTemplate, scalar) require an operator from {+, -, *, /}')
            scalar = rhs = self._parse_operand(rhs, lhs.defined_channels)
            pulse_template = lhs

        else:
            # + and - with (AtomicPulseTemplate, AtomicPulseTemplate) as operands
            raise TypeError('ArithmeticPulseTemplate cannot combine two PulseTemplates')

        self._lhs = lhs
        self._rhs = rhs

        self._pulse_template = pulse_template
        self._scalar = scalar

        self._arithmetic_operator = arithmetic_operator

    @staticmethod
    def _parse_operand(operand: Union[ExpressionLike, Mapping[ChannelID, ExpressionLike]],
                       channels: Set[ChannelID]) -> Union[ExpressionScalar, Mapping[ChannelID, ExpressionScalar]]:
        """Transforms operand or all entries of operand to ExpressionScalar

        Args:
            operand: operands to transforms
            channels: Guard against non defined channels

        Raises:
            ValueError if a channel is in the operand that is not in channels

        Returns:
            A dict with ExpressionScalar values or an ExpressionScalar
        """
        if isinstance(operand, Mapping):
            missing_in_channels = operand.keys() - channels
            if missing_in_channels:
                raise ValueError('The channels {} are defined in the operand but not in the pulse template.'.format(
                    missing_in_channels))
            operand = {channel: value if isinstance(value, ExpressionScalar) else ExpressionScalar(value)
                       for channel, value in operand.items()}
            return operand
        else:
            return operand if isinstance(operand, ExpressionScalar) else ExpressionScalar(operand)

    def _get_scalar_value(self,
                          parameters: Mapping[str, Real],
                          channel_mapping: Mapping[str, Optional[str]]) -> Dict[ChannelID, Real]:
        """Generate a dict of real values from the scalar operand.

        If the scalar operand is an ExpressionScalar all channels with non None values in channel_mapping get the same
        output.

        If the scalar operand is a Mapping only those mapped to non None are in the output

        Args:
            parameters:
            channel_mapping:

        Returns:
            The evaluation of the scalar operand for all relevant channels
        """
        if isinstance(self._scalar, ExpressionScalar):
            scalar_value = self._scalar.evaluate_in_scope(parameters)
            return {channel_mapping[channel]: scalar_value
                    for channel in self._pulse_template.defined_channels
                    if channel_mapping[channel]}

        else:
            return {channel_mapping[channel]: value.evaluate_in_scope(parameters)
                    for channel, value in self._scalar.items()
                    if channel_mapping[channel]}

    @property
    def lhs(self):
        return self._lhs

    @property
    def rhs(self):
        return self._rhs
    
    def _get_transformation(self,
                            parameters: Mapping[str, Real],
                            channel_mapping: Mapping[ChannelID, ChannelID]) -> Transformation:
        transformation = IdentityTransformation()

        scalar_value = self._get_scalar_value(parameters=parameters,
                                              channel_mapping=channel_mapping)

        if self._pulse_template is self._rhs:
            if self._arithmetic_operator == '-':
                # negate the pulse template
                transformation = transformation.chain(
                    ScalingTransformation({channel_mapping[ch]: -1
                                           for ch in self.defined_channels
                                           if channel_mapping[ch]}))

        else:
            if self._arithmetic_operator == '-':
                for channel, value in scalar_value.items():
                    scalar_value[channel] = -value

            elif self._arithmetic_operator == '/':
                for channel, value in scalar_value.items():
                    scalar_value[channel] = 1/value

        if self._arithmetic_operator in ('+', '-'):
            return transformation.chain(
                OffsetTransformation(scalar_value)
            )

        else:
            return transformation.chain(
                ScalingTransformation(scalar_value)
            )

    def _internal_create_program(self, *,
                                 scope: Scope,
                                 measurement_mapping: Dict[str, Optional[str]],
                                 channel_mapping: Dict[ChannelID, Optional[ChannelID]],
                                 global_transformation: Optional[Transformation],
                                 to_single_waveform: Set[Union[str, 'PulseTemplate']],
                                 parent_loop: 'Loop'):
        """The operation is applied by modifying the transformation the pulse template operand sees."""
        if not scope.get_volatile_parameters().keys().isdisjoint(self._scalar_operand_parameters):
            raise NotImplementedError('The scalar operand of arithmetic pulse template cannot be volatile')

        # put arithmetic into transformation
        inner_transformation = self._get_transformation(parameters=scope,
                                                        channel_mapping=channel_mapping)

        transformation = inner_transformation.chain(global_transformation)

        self._pulse_template._create_program(scope=scope,
                                             measurement_mapping=measurement_mapping,
                                             channel_mapping=channel_mapping,
                                             global_transformation=transformation,
                                             to_single_waveform=to_single_waveform,
                                             parent_loop=parent_loop)

    def build_waveform(self,
                       parameters: Dict[str, Real],
                       channel_mapping: Dict[ChannelID, ChannelID]) -> Optional[Waveform]:
        pt = cast(AtomicPulseTemplate, self._pulse_template)
        inner_waveform = pt.build_waveform(parameters=parameters, channel_mapping=channel_mapping)

        if inner_waveform is None:
            return None

        # put arithmetic into transformation
        transformation = self._get_transformation(parameters=parameters,
                                                  channel_mapping=channel_mapping)

        return TransformingWaveform(inner_waveform, transformation=transformation)

    def __repr__(self):
        if any(v for k, v in super().get_serialization_data().items() if k != '#type'):
            return super().__repr__()
        else:
            return '(%r %s %r)' % (self.lhs, self._arithmetic_operator, self.rhs)

    def get_serialization_data(self, serializer: Optional['Serializer'] = None) -> Dict:
        if serializer:
            raise NotImplementedError('Compatibility to old serialization routines not implemented for new type')

        data = super().get_serialization_data()

        data['rhs'] = self.rhs
        data['lhs'] = self.lhs
        data['arithmetic_operator'] = self._arithmetic_operator

        return data

    @property
    def defined_channels(self):
        return self._pulse_template.defined_channels

    @property
    def duration(self) -> ExpressionScalar:
        return self._pulse_template.duration

    @property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        integral = {channel: value.sympified_expression for channel, value in self._pulse_template.integral.items()}

        if isinstance(self._scalar, ExpressionScalar):
            scalar = {channel: self._scalar.sympified_expression
                      for channel in self.defined_channels}
        else:
            scalar = {channel: value.sympified_expression
                      for channel, value in self._scalar.items()}

        if self._arithmetic_operator == '+':
            for channel, value in scalar.items():
                integral[channel] = integral[channel] + (value * self.duration.sympified_expression)

        elif self._arithmetic_operator == '*':
            for channel, value in scalar.items():
                integral[channel] = integral[channel] * value

        elif self._arithmetic_operator == '/':
            assert self._pulse_template is self.lhs
            for channel, value in scalar.items():
                integral[channel] = integral[channel] / value

        else:
            assert self._arithmetic_operator == '-'
            if self._pulse_template is self.rhs:
                # we need to negate all existing values
                for channel, inner_value in integral.items():
                    if channel in scalar:
                        integral[channel] = scalar[channel] * self.duration.sympified_expression - inner_value
                    else:
                        integral[channel] = -inner_value

            else:
                for channel, value in scalar.items():
                    integral[channel] = integral[channel] - value * self.duration.sympified_expression

        for channel, value in integral.items():
            integral[channel] = ExpressionScalar(value)
        return integral

    @property
    def measurement_names(self) -> Set[str]:
        return self._pulse_template.measurement_names

    @cached_property.cached_property
    def _scalar_operand_parameters(self) -> FrozenSet[str]:
        if isinstance(self._scalar, dict):
            return frozenset(*(value.variables for value in self._scalar.values()))
        else:
            return frozenset(self._scalar.variables)

    @property
    def parameter_names(self) -> Set[str]:
        return self._pulse_template.parameter_names.union(self._scalar_operand_parameters)


def try_operation(lhs: Union[PulseTemplate, ExpressionLike, Mapping[ChannelID, ExpressionLike]],
                  op: str,
                  rhs: Union[PulseTemplate, ExpressionLike, Mapping[ChannelID, ExpressionLike]],
                  **kwargs) -> Union['ArithmeticPulseTemplate', type(NotImplemented)]:
    """

    Args:
        lhs: Left hand side operand
        op: String representation of the operator
        rhs: Right hand side operand
        **kwargs: Forwarded to class init

    Returns:
        ArithmeticPulseTemplate if the desired operation is valid and returns a pulse template
        NotImplemented otherwise
    """
    try:
        # returns if only one of the operands is a pulse template and the operation is valid
        return ArithmeticPulseTemplate(lhs, op, rhs, **kwargs)
    except TypeError:
        # either none or both are pulse templates
        try:
            return ArithmeticAtomicPulseTemplate(lhs, op, rhs, **kwargs)
        except ValueError:
            # invalid operand
            return NotImplemented
    except ValueError:
        # invalid operand
        return NotImplemented


class UnequalDurationWarningInArithmeticPT(RuntimeWarning):
    """Signals that an ArithmeticAtomicPulseTemplate was constructed from operands with unequal duration. This is a
    separate class to allow easy silencing."""


class ImplicitAtomicityInArithmeticPT(RuntimeWarning):
    """Signals that an ArithmeticAtomicPulseTemplate has operands that are non-atomic but will be interpreted as atomic.
    This is a separate class to allow easy silencing.
    """
