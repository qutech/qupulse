"""This module defines the FunctionPulseTemplate, one of the elementary pulse templates and its
waveform representation.

Classes:
    - FunctionPulseTemplate: Defines a pulse via a mathematical function.
    - FunctionWaveform: A waveform instantiated from a FunctionPulseTable.
"""


from typing import Any, Dict, List, Set, Optional, Union
import numbers
import itertools

import numpy as np

from qctoolkit.expressions import Expression
from qctoolkit.serialization import Serializer

from qctoolkit import MeasurementWindow, ChannelID
from qctoolkit.pulses.parameters import ParameterDeclaration, Parameter
from qctoolkit.pulses.pulse_template import AtomicPulseTemplate, MeasurementDeclaration
from qctoolkit.pulses.instructions import Waveform
from qctoolkit.pulses.pulse_template_parameter_mapping import ParameterNotProvidedException


__all__ = ["FunctionPulseTemplate", "FunctionWaveform"]


class FunctionPulseTemplate(AtomicPulseTemplate):
    """Defines a pulse via a time-domain expression.

    FunctionPulseTemplate stores the expression and its external parameters. The user must provide
    two things: one expression that calculates the length of the pulse from the external parameters
    and the time-domain pulse shape itself as a expression. The required external parameters are
    derived from the free variables in the expressions themselves.
    Like other PulseTemplates the FunctionPulseTemplate can be declared to be a measurement pulse.

    The independent variable for the time domain in the expression is expected to be called 't'.
    """

    def __init__(self,
                 expression: Union[str, Expression],
                 duration_expression: Union[str, Expression],
                 measurements: Optional[List[MeasurementDeclaration]]=None,
                 identifier: str=None,
                 channel: 'ChannelID' = 'default') -> None:
        """Create a new FunctionPulseTemplate instance.

        Args:
            expression (str or Expression): The function represented by this FunctionPulseTemplate
                as a mathematical expression where 't' denotes the time variable and other variables
                will be parameters of the pulse.
            duration_expression (str or Expression): A mathematical expression which reliably
                computes the duration of an instantiation of this FunctionPulseTemplate from
                provided parameter values.
            measurement (bool): True, if this FunctionPulseTemplate shall define a measurement
                window. (optional, default = False)
            identifier (str): A unique identifier for use in serialization. (optional)
        """
        super().__init__(identifier=identifier, measurements=measurements)
        self.__expression = expression
        if not isinstance(self.__expression, Expression):
            self.__expression = Expression(self.__expression)
        self.__duration_expression = duration_expression
        if not isinstance(self.__duration_expression, Expression):
            self.__duration_expression = Expression(self.__duration_expression)
        self.__parameter_names = set(self.__duration_expression.variables
                                     + self.__expression.variables) - set(['t'])
        self.__channel = channel
        self.__measurement_windows = dict()

    @property
    def function_parameters(self) -> Set[str]:
        return self.__parameter_names

    @property
    def parameter_names(self) -> Set[str]:
        return self.function_parameters | self.measurement_parameters

    @property
    def parameter_declarations(self) -> Set[ParameterDeclaration]:
        return {ParameterDeclaration(param_name) for param_name in self.parameter_names}

    @property
    def is_interruptable(self) -> bool:
        return False

    @property
    def defined_channels(self) -> Set['ChannelID']:
        return {self.__channel}

    @property
    def duration(self) -> Expression:
        return self.__duration_expression

    def build_waveform(self,
                       parameters: Dict[str, Parameter],
                       measurement_mapping: Dict[str, str],
                       channel_mapping: Dict[ChannelID, ChannelID]) -> 'FunctionWaveform':
        substitutions = dict((v, parameters[v].get_value()) for v in self.__expression.variables if v != 't')
        duration_parameters = dict((v, parameters[v].get_value()) for v in self.__duration_expression.variables)
        return FunctionWaveform(expression=self.__expression.evaluate_symbolic(substitutions=substitutions),
                                duration=self.__duration_expression.evaluate_numeric(**duration_parameters),
                                measurement_windows=self.get_measurement_windows(parameters=parameters,
                                                                                 measurement_mapping=measurement_mapping),
                                channel=channel_mapping[self.__channel])

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, 'Condition']) -> bool:
        return any(
            parameters[name].requires_stop
            for name in parameters.keys() if (name in self.parameter_names)
        )

    def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
        return dict(
            duration_expression=serializer.dictify(self.__duration_expression),
            expression=serializer.dictify(self.__expression),
            channel=self.__channel,
            measurement_declarations=self.measurement_declarations
        )

    @staticmethod
    def deserialize(serializer: 'Serializer',
                    expression: str,
                    duration_expression: str,
                    channel: 'ChannelID',
                    measurement_declarations: Dict[str, List],
                    identifier: Optional[bool]=None) -> 'FunctionPulseTemplate':
        return FunctionPulseTemplate(
            serializer.deserialize(expression),
            serializer.deserialize(duration_expression),
            channel=channel,
            identifier=identifier,
            measurements=measurement_declarations
        )


class FunctionWaveform(Waveform):
    """Waveform obtained from instantiating a FunctionPulseTemplate."""

    def __init__(self, expression: Expression,
                 duration: float,
                 measurement_windows: List[MeasurementWindow],
                 channel: ChannelID) -> None:
        """Creates a new FunctionWaveform instance.

        Args:
            expression (Expression): The function represented by this FunctionWaveform
                as a mathematical expression where 't' denotes the time variable. It may not have other variables
            duration (float): A mathematical expression which reliably
                computes the duration of this FunctionPulseTemplate.
            measurement_windows (List): A list of measurement windows
        """
        super().__init__()
        if set(expression.variables) - set('t'):
            raise ValueError('FunctionWaveforms may not depend on anything but "t"')

        self._expression = expression
        self._duration = duration
        self._channel_id = channel
        self._measurement_windows = measurement_windows

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return {self._channel_id}

    def get_measurement_windows(self) -> List[MeasurementWindow]:
        return self._measurement_windows
    
    @property
    def compare_key(self) -> Any:
        return self._channel_id, self._expression, self._duration, self._measurement_windows

    @property
    def duration(self) -> float:
        return self._duration

    def unsafe_sample(self,
                      channel: ChannelID,
                      sample_times: np.ndarray,
                      output_array: Union[np.ndarray, None] = None) -> np.ndarray:
        if output_array is None:
            output_array = np.empty(len(sample_times))
        output_array[:] = self._expression.evaluate_numeric(t=sample_times)
        return output_array

    def unsafe_get_subset_for_channels(self, channels: Set[ChannelID]) -> Waveform:
        return self
