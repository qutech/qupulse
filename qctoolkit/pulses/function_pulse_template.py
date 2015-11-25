import logging
import numbers

import numpy as np
from typing import Any
from typing import Dict, Set

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from qctoolkit.expressions import Expression
from qctoolkit.serialization import Serializer

from .parameters import ParameterDeclaration, Parameter
from .pulse_template import PulseTemplate
from .measurements import Measurement
from .sequencing import InstructionBlock, Sequencer
from .sequence_pulse_template import ParameterNotProvidedException
from .instructions import Waveform

logger = logging.getLogger(__name__)

__all__ = ["FunctionPulseTemplate", "FunctionWaveform"]


class FunctionPulseTemplate(PulseTemplate):
    """Defines a pulse via a time-domain expression.

    FunctionPulseTemplate stores the expression and its external parameters. The user must provide
    two things: one expression that calculates the length of the pulse from the external parameters
    and the time-domain pulse shape itself as a expression. The external parameters are derived from
    the expressions themselves.
    Like other PulseTemplates the FunctionPulseTemplate can be declared to be a measurement pulse.

    The independent variable in the expression is called 't' and is given in units of nano-seconds.
    """

    def __init__(self, expression: str, duration_expression: str, measurement: bool=False) -> None:
        super().__init__()
        self.__expression = Expression(expression)
        self.__duration_expression = Expression(duration_expression)
        self.__is_measurement_pulse = measurement # type: bool
        self.__parameter_names = set(self.__duration_expression.variables() + self.__expression.variables()) - set(['t'])
        self.__measurement = Measurement(self)

    @property
    def parameter_names(self) -> Set[str]:
        """Return the set of names of declared parameters."""
        return self.__parameter_names

    @property
    def parameter_declarations(self) -> Set[ParameterDeclaration]:
        """Return a set of all parameter declaration objects of this TablePulseTemplate."""
        return set()

    def get_pulse_length(self, parameters) -> float:
        """Return the length of this pulse for the given parameters."""
        missing = self.__parameter_names - set(parameters.keys())
        for m in missing:
            raise ParameterNotProvidedException(m)
        return self.__duration_expression.evaluate(**parameters)

    @property
    def is_interruptable(self) -> bool:
        """Return true, if this PulseTemplate contains points at which it can halt if interrupted."""
        return False
        
    def build_sequence(self,
                       sequencer: Sequencer,
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, 'Condition'],
                       instruction_block: InstructionBlock) -> None:
        if self.__is_measurement_pulse:
            self.__measurement.measure(self.get_pulse_length(parameters))
        waveform = FunctionWaveform(parameters, self.__expression, self.__duration_expression, self.__measurement.instantiate(parameters))
        instruction_block.add_instruction_exec(waveform)

    def requires_stop(self, parameters: Dict[str, Parameter], conditions: Dict[str, 'Condition']) -> bool:
        return any(parameters[name].requires_stop for name in parameters.keys() if (name in self.parameter_names) and not isinstance(parameters[name], numbers.Number))

    def get_serialization_data(self, serializer: Serializer) -> None:
        root = dict()
        root['type'] = 'FunctionPulseTemplate'
        root['parameter_names'] = self.__parameter_names
        root['duration_expression'] = self.__duration_expression.string
        root['expression'] = self.__expression.string
        root['measurement'] = self.__is_measurement_pulse
        return root

    @staticmethod
    def deserialize(serializer: 'Serializer', **kwargs) -> 'Serializable':
        return FunctionPulseTemplate(kwargs['expression'], kwargs['duration_expression'], kwargs['Measurement'])


class FunctionWaveform(Waveform):
    def __init__(self, parameters: Dict[str, Parameter], expression: Expression, duration_expression: Expression,
                 measurement: Measurement = None) -> None:
        super().__init__()
        self.__expression = expression
        self.__parameters = parameters
        self.__duration_expression = duration_expression
        self.__partial_expression = self.__partial
        self.__measurement = measurement
    
    def __partial (self,t):
        params = self.__parameters.copy()
        params.update({"t":t})
        return self.__expression.evaluate(**params)
    
    @property
    def _compare_key(self) -> Any:
        return self.__expression

    @property
    def duration(self) -> float:
        return self.__duration_expression.evaluate(**self.__parameters)

    def sample(self, sample_times: np.ndarray, first_offset: float=0) -> np.ndarray:
        sample_times -= (sample_times[0] - first_offset)
        func = np.vectorize(self.__partial_expression)
        voltages = func(sample_times)
        return voltages

    @property
    def measurement_windows(self, first_offset: float = 0):
        self.__measurement.offset = first_offset
        return self.__measurement.build()
