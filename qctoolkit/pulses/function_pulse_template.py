"""This module defines the FunctionPulseTemplate, one of the elementary pulse templates and its
waveform representation.

Classes:
    - FunctionPulseTemplate: Defines a pulse via a mathematical function.
    - FunctionWaveform: A waveform instantiated from a FunctionPulseTable.
"""


from typing import Any, Dict, List, Set, Optional, Union

import numpy as np

from qctoolkit.expressions import Expression
from qctoolkit.serialization import Serializer

from qctoolkit import MeasurementWindow, ChannelID
from qctoolkit.pulses.parameters import ParameterDeclaration, Parameter
from qctoolkit.pulses.pulse_template import AtomicPulseTemplate
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
        super().__init__(identifier)
        self.__expression = expression
        if not isinstance(self.__expression, Expression):
            self.__expression = Expression(self.__expression)
        self.__duration_expression = duration_expression
        if not isinstance(self.__duration_expression, Expression):
            self.__duration_expression = Expression(self.__duration_expression)
        self.__parameter_names = set(self.__duration_expression.variables()
                                     + self.__expression.variables()) - set(['t'])
        self.__channel = channel

    @property
    def parameter_names(self) -> Set[str]:
        return self.__parameter_names

    @property
    def parameter_declarations(self) -> Set[ParameterDeclaration]:
        return {ParameterDeclaration(param_name) for param_name in self.parameter_names}

    @property
    def measurement_names(self) -> Set[str]:
        return set()

    @property
    def is_interruptable(self) -> bool:
        return False

    @property
    def defined_channels(self) -> Set['ChannelID']:
        return set(self.__channel)

    def build_waveform(self,
                       parameters: Dict[str, Parameter],
                       measurement_mapping: Dict[str, str],
                       channel_mapping: Dict[ChannelID, ChannelID]) -> 'FunctionWaveform':
        return FunctionWaveform(
            channel=channel_mapping[self.__channel],
            parameters={parameter_name: parameter.get_value()
                        for (parameter_name, parameter) in parameters.items()},
            expression=self.__expression,
            duration_expression=self.__duration_expression,
            measurement_windows=[]
        )

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, 'Condition']) -> bool:
        return any(
            parameters[name].requires_stop
            for name in parameters.keys() if (name in self.parameter_names)
        )

    def get_serialization_data(self, serializer: Serializer) -> None:
        return dict(
            duration_expression=serializer.dictify(self.__duration_expression),
            expression=serializer.dictify(self.__expression),
            channel=self.__channel
        )

    @staticmethod
    def deserialize(serializer: 'Serializer',
                    expression: str,
                    duration_expression: str,
                    channel: 'ChannelID',
                    identifier: Optional[bool]=None) -> 'FunctionPulseTemplate':
        return FunctionPulseTemplate(
            serializer.deserialize(expression),
            serializer.deserialize(duration_expression),
            channel=channel,
            identifier=identifier
        )


class FunctionWaveform(Waveform):
    """Waveform obtained from instantiating a FunctionPulseTemplate."""

    def __init__(self,
                 parameters: Dict[str, float],
                 expression: Expression,
                 duration_expression: Expression,
                 measurement_windows: List[MeasurementWindow],
                 channel: ChannelID
                 ) -> None:
        """Creates a new FunctionWaveform instance.

        Args:
            parameters (Dict(str -> float)): A mapping of parameter names to parameter values.
            expression (Expression): The function represented by this FunctionWaveform
                as a mathematical expression where 't' denotes the time variable and other variables
                are filled with values from the parameters mapping.
            duration_expression (Expression): A mathematical expression which reliably
                computes the duration of this FunctionPulseTemplate.
        """
        super().__init__()
        self.__expression = expression
        self.__parameters = parameters
        self.__duration = duration_expression.evaluate_numeric(**self.__parameters)
        self.__channel_id = channel
        self.__measurement_windows = measurement_windows

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return {self.__channel_id}

    def get_measurement_windows(self) -> List[MeasurementWindow]:
        return self.__measurement_windows
    
    def __evaluate(self, t) -> float:
        params = self.__parameters.copy()
        params.update({"t": t})
        return self.__expression.evaluate_numeric(**params)
    
    @property
    def compare_key(self) -> Any:
        return self.__channel_id, self.__expression, self.__duration, self.__parameters

    @property
    def duration(self) -> float:
        return self.__duration

    def unsafe_sample(self,
                      channel: ChannelID,
                      sample_times: np.ndarray,
                      output_array: Union[np.ndarray, None] = None) -> np.ndarray:
        if output_array is None:
            output_array = np.empty(len(sample_times))
        output_array[:] = self.__evaluate(sample_times)
        return output_array

    def unsafe_get_subset_for_channels(self, channels: Set[ChannelID]) -> Waveform:
        return self
