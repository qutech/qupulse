"""This module defines the FunctionPulseTemplate, one of the elementary pulse templates and its
waveform representation.

Classes:
    - FunctionPulseTemplate: Defines a pulse via a mathematical function.
"""


from typing import Any, Dict, List, Set, Optional, Union
import numbers

import sympy

from qupulse.expressions import ExpressionScalar
from qupulse.serialization import Serializer, PulseRegistryType

from qupulse.utils.types import ChannelID, TimeType, time_from_float
from qupulse.pulses.parameters import Parameter, ParameterConstrainer, ParameterConstraint
from qupulse.pulses.pulse_template import AtomicPulseTemplate, MeasurementDeclaration
from qupulse._program.waveforms import FunctionWaveform


__all__ = ["FunctionPulseTemplate"]


class FunctionPulseTemplate(AtomicPulseTemplate, ParameterConstrainer):
    """Defines a pulse via a time-domain expression.

    FunctionPulseTemplate stores the expression and its external parameters. The user must provide
    two things: one expression that calculates the length of the pulse from the external parameters
    and the time-domain pulse shape itself as a expression. The required external parameters are
    derived from the free variables in the expressions themselves.
    Like other PulseTemplates the FunctionPulseTemplate can be declared to be a measurement pulse.

    The independent variable for the time domain in the expression is expected to be called 't'.
    """

    def __init__(self,
                 expression: Union[str, ExpressionScalar],
                 duration_expression: Union[str, ExpressionScalar],
                 channel: ChannelID = 'default',
                 identifier: Optional[str] = None,
                 *,
                 measurements: Optional[List[MeasurementDeclaration]]=None,
                 parameter_constraints: Optional[List[Union[str, ParameterConstraint]]]=None,
                 registry: PulseRegistryType=None) -> None:
        """Creates a new FunctionPulseTemplate object.

        Args:
            expression: The function represented by this FunctionPulseTemplate
                as a mathematical expression where 't' denotes the time variable and other variables
                will be parameters of the pulse.
            duration_expression: A mathematical expression which reliably
                computes the duration of an instantiation of this FunctionPulseTemplate from
                provided parameter values.
            channel: The channel this pulse template is defined on.
            identifier: A unique identifier for use in serialization.
            measurements: A list of measurement declarations forwarded to the
                :class:`~qupulse.pulses.measurement.MeasurementDefiner` superclass
            parameter_constraints: A list of parameter constraints forwarded to the
                :class:`~qupulse.pulses.measurement.ParameterConstrainer` superclass
        """
        AtomicPulseTemplate.__init__(self, identifier=identifier, measurements=measurements)
        ParameterConstrainer.__init__(self, parameter_constraints=parameter_constraints)

        self.__expression = ExpressionScalar.make(expression)
        self.__duration_expression = ExpressionScalar.make(duration_expression)
        self.__parameter_names = {*self.__duration_expression.variables, *self.__expression.variables} - {'t'}
        self.__channel = channel

        self._register(registry=registry)

    @property
    def expression(self) -> ExpressionScalar:
        return self.__expression

    @property
    def function_parameters(self) -> Set[str]:
        return self.__parameter_names

    @property
    def parameter_names(self) -> Set[str]:
        return self.function_parameters | self.measurement_parameters | self.constrained_parameters

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return {self.__channel}

    @property
    def duration(self) -> ExpressionScalar:
        return self.__duration_expression

    def build_waveform(self,
                       parameters: Dict[str, numbers.Real],
                       channel_mapping: Dict[ChannelID, Optional[ChannelID]]) -> Optional['FunctionWaveform']:
        self.validate_parameter_constraints(parameters=parameters, volatile=set())

        channel = channel_mapping[self.__channel]
        if channel is None:
            return None

        if 't' in parameters:
            parameters = {k: v for k, v in parameters.items() if k != 't'}

        expression = self.__expression.evaluate_symbolic(substitutions=parameters)
        duration = self.__duration_expression.evaluate_with_exact_rationals(parameters)

        return FunctionWaveform.from_expression(expression=expression,
                                duration=duration,
                                channel=channel_mapping[self.__channel])

    def get_serialization_data(self, serializer: Optional[Serializer]=None) -> Dict[str, Any]:
        data = super().get_serialization_data(serializer)

        if serializer: # compatibility to old serialization routines, deprecated
            return dict(
                duration_expression=self.__duration_expression,
                expression=self.__expression,
                channel=self.__channel,
                measurement_declarations=self.measurement_declarations,
                parameter_constraints=[str(c) for c in self.parameter_constraints]
            )

        local_data = dict(
            duration_expression=self.__duration_expression,
            expression=self.__expression,
            channel=self.__channel,
            measurements=self.measurement_declarations,
            parameter_constraints=[str(c) for c in self.parameter_constraints]
        )

        data.update(**local_data)
        return data

    @classmethod
    def deserialize(cls,
                    serializer: Optional[Serializer]=None,
                    **kwargs) -> 'FunctionPulseTemplate':
        if serializer:
            kwargs['measurements'] = kwargs['measurement_declarations'] # compatibility to old serialization routines, deprecated
            del kwargs['measurement_declarations']
        return super().deserialize(None, **kwargs)

    @property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        return {self.__channel: ExpressionScalar(
            sympy.integrate(self.__expression.sympified_expression, ('t', 0, self.duration.sympified_expression))
        )}

    def _as_expression(self) -> Dict[ChannelID, ExpressionScalar]:
        expr = ExpressionScalar.make(self.__expression.underlying_expression.subs({'t': self._AS_EXPRESSION_TIME}))
        return {self.__channel: expr}

    @property
    def initial_values(self) -> Dict[ChannelID, ExpressionScalar]:
        expr = ExpressionScalar.make(self.__expression.underlying_expression.subs('t', 0))
        return {self.__channel: expr}

    @property
    def final_values(self) -> Dict[ChannelID, ExpressionScalar]:
        expr = ExpressionScalar.make(self.__expression.underlying_expression.subs('t', self.__duration_expression.underlying_expression))
        return {self.__channel: expr}


