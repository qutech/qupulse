import numbers
from typing import (Any, Callable, Dict, List, Optional, Set, Union)

import numpy as np
from qupulse._program.waveforms import MethodWaveform
from qupulse.expressions import ExpressionScalar
from qupulse.pulses.measurement import MeasurementDeclaration
from qupulse.pulses.parameters import ParameterConstrainer, ParameterConstraint
from qupulse.pulses.pulse_template import (AtomicPulseTemplate, ChannelID)
from qupulse.serialization import PulseRegistryType, Serializer


import functools
@functools.lru_cache
def ExpressionScalarCache(value):
    return ExpressionScalar(value)
        
class MethodPulseTemplate(AtomicPulseTemplate, ParameterConstrainer):
    """Defines a pulse via a method

    MethodPulseTemplate.

    """

    def __init__(self,
                 pulse_method: Callable,
                 duration: ExpressionScalar,
                 channel: ChannelID = 'default',
                 identifier: Optional[str] = None,
                 *,
                 measurements: Optional[List[MeasurementDeclaration]] = None,
                 parameter_constraints: Optional[List[Union[str, ParameterConstraint]]] = None,
                 registry: PulseRegistryType = None) -> None:
        """Creates a new FunctionPulseTemplate object.

        Args:
            method: The function represented by this MethodPulseTemplate
            duration: Duration
            channel: The channel this pulse template is defined on.
            identifier: A unique identifier for use in serialization.
            measurements: A list of measurement declarations forwarded to the
                :class:`~qupulse.pulses.measurement.MeasurementDefiner` superclass
            parameter_constraints: A list of parameter constraints forwarded to the
                :class:`~qupulse.pulses.measurement.ParameterConstrainer` superclass
        """
        AtomicPulseTemplate.__init__(self, identifier=identifier, measurements=measurements)
        ParameterConstrainer.__init__(self, parameter_constraints=parameter_constraints)

        self._pulse_method = pulse_method
        self._duration = ExpressionScalarCache(duration)
        self.__parameter_names: Set[str] = set()
        self.__channel = channel

        self._register(registry=registry)

    def _as_expression(self) -> Dict[ChannelID, ExpressionScalar]:
        raise NotImplementedError(f'expression not available for {self.__class__}')

    @property
    def pulse_method(self) -> Callable:
        return self._pulse_method

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
        return self._duration

    def build_waveform(self,
                       parameters: Dict[str, numbers.Real],
                       channel_mapping: Dict[ChannelID, Optional[ChannelID]]) -> Optional['MethodWaveform']:
        self.validate_parameter_constraints(parameters=parameters, volatile=set())

        channel = channel_mapping[self.__channel]
        if channel is None:
            return None

        duration = self._duration

        return MethodWaveform(pulse_method=self.pulse_method,
                              duration=float(duration),
                              channel=channel_mapping[self.__channel])

    def get_serialization_data(self, serializer: Optional[Serializer] = None) -> Dict[str, Any]:
        data = super().get_serialization_data(serializer)

        if serializer:  # compatibility to old serialization routines, deprecated
            raise NotImplementedError

        local_data = dict(
            duration=self.duration,
            method=str(self.pulse_method),
            channel=self.__channel,
            measurements=self.measurement_declarations,
            parameter_constraints=[str(c) for c in self.parameter_constraints]
        )

        data.update(**local_data)
        return data

    @classmethod
    def deserialize(cls,
                    serializer: Optional[Serializer] = None,
                    **kwargs) -> 'MethodPulseTemplate':
        raise NotImplementedError()

    @property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        try:
            import scipy.integrate
        except ImportError:
            raise ValueError(f'scipy package is required to perform integral calculations for {self.__class__}')
                
        return {self.__channel: ExpressionScalar(scipy.integrate.quad(self._pulse_method, 0, float(self.duration))[0]
                                                 )}



if __name__ == '__main__':
    from qupulse.pulses.plotting import plot
    px = MethodPulseTemplate(pulse_method=lambda t: np.sin(.2*t), duration=100)
    plot(px, sample_rate=10)
