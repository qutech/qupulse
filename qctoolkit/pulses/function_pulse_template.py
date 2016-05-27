"""This module defines the FunctionPulseTemplate, one of the elementary pulse templates and its
waveform representation.

Classes:
    - FunctionPulseTemplate: Defines a pulse via a mathematical function.
    - FunctionWaveform: A waveform instantiated from a FunctionPulseTable.
"""


from typing import Any, Dict, List, Set, Optional, Union
import numbers

import numpy as np

from qctoolkit.expressions import Expression
from qctoolkit.serialization import Serializer

from qctoolkit.pulses.parameters import ParameterDeclaration, Parameter
from qctoolkit.pulses.pulse_template import PulseTemplate, MeasurementWindow
from qctoolkit.pulses.sequencing import InstructionBlock, Sequencer
from qctoolkit.pulses.sequence_pulse_template import ParameterNotProvidedException
from qctoolkit.pulses.instructions import Waveform

__all__ = ["FunctionPulseTemplate", "FunctionWaveform"]


class FunctionPulseTemplate(PulseTemplate):
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
                 measurement: bool=False,
                 identifier: str=None) -> None:
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
        self.__is_measurement_pulse = measurement # type: bool
        self.__parameter_names = set(self.__duration_expression.variables()
                                     + self.__expression.variables()) - set(['t'])

    @property
    def parameter_names(self) -> Set[str]:
        return self.__parameter_names

    @property
    def parameter_declarations(self) -> Set[ParameterDeclaration]:
        return [ParameterDeclaration(param_name) for param_name in self.parameter_names]

    def get_pulse_length(self, parameters: Dict[str, Parameter]) -> float:
        """Return the length of this pulse for the given parameters.

        OBSOLETE/FLAWED? Just used in by get_measurement_windows which days are counted.

        Args:
            parameters (Dict(str -> Parameter)): A mapping of parameter name to parameter objects.
        """
        missing_parameters = self.__parameter_names - set(parameters.keys())
        for missing_parameter in missing_parameters:
            raise ParameterNotProvidedException(missing_parameter)
        return self.__duration_expression.evaluate(
            **{parameter_name: parameter.get_value()
               for (parameter_name, parameter) in parameters.items()}
        )

    def get_measurement_windows(self,
                                parameters: Optional[Dict[str, Parameter]]=None) \
            -> List[MeasurementWindow]:
        raise NotImplementedError()
        if not self.__is_measurement_pulse:
            return
        else:
            return [(0, self.get_pulse_length(parameters))]

    @property
    def is_interruptable(self) -> bool:
        return False
        
    def build_sequence(self,
                       sequencer: Sequencer,
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, 'Condition'],
                       instruction_block: InstructionBlock) -> None:
        waveform = FunctionWaveform(
            {parameter_name: parameter.get_value()
             for (parameter_name, parameter) in parameters.items()},
            self.__expression,
            self.__duration_expression
        )
        instruction_block.add_instruction_exec(waveform)

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, 'Condition']) -> bool:
        return any(
            parameters[name].requires_stop
            for name in parameters.keys()
            if (name in self.parameter_names) and not isinstance(parameters[name], numbers.Number)
        )

    def get_serialization_data(self, serializer: Serializer) -> None:
        root = dict()
        root['type'] = 'FunctionPulseTemplate'
        root['parameter_names'] = self.__parameter_names
        root['duration_expression'] = serializer.dictify(self.__duration_expression)
        root['expression'] = serializer.dictify(self.__expression)
        root['measurement'] = self.__is_measurement_pulse
        return root

    @staticmethod
    def deserialize(serializer: 'Serializer', **kwargs) -> 'Serializable':
        return FunctionPulseTemplate(
            kwargs['expression'],
            kwargs['duration_expression'],
            kwargs['Measurement']
        )


class FunctionWaveform(Waveform):
    """Waveform obtained from instantiating a FunctionPulseTemplate."""

    def __init__(self,
                 parameters: Dict[str, float],
                 expression: Expression,
                 duration_expression: Expression) -> None:
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
        self.__duration = duration_expression.evaluate(**self.__parameters)
    
    def __evaluate_partially(self, t):
        params = self.__parameters.copy()
        params.update({"t":t})
        return self.__expression.evaluate(**params)
    
    @property
    def compare_key(self) -> Any:
        return self.__expression

    @property
    def duration(self) -> float:
        return self.__duration

    def sample(self, sample_times: np.ndarray, first_offset: float=0) -> np.ndarray:
        sample_times -= (sample_times[0] - first_offset)
        func = np.vectorize(self.__evaluate_partially)
        voltages = func(sample_times)
        return voltages
