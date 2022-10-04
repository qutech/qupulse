"""This module defines the ConstantPulseTemplate, a pulse tempalte representating a pulse with constant values on all channels

Classes:
    - ConstantPulseTemplate: Defines a pulse via channel-value pairs
    - ConstantPulseWaveform: A waveform instantiated from a TablePulseTemplate by providing values for its
        declared parameters.
"""

import logging
import numbers
from typing import Any, Dict, List, Optional, Union, Mapping, AbstractSet

from qupulse._program.waveforms import ConstantWaveform
from qupulse.utils.types import TimeType, ChannelID
from qupulse.utils import cached_property
from qupulse.expressions import ExpressionScalar, ExpressionLike
from qupulse.pulses.multi_channel_pulse_template import MultiChannelWaveform
from qupulse.pulses.pulse_template import AtomicPulseTemplate, MeasurementDeclaration
from qupulse.serialization import PulseRegistryType

__all__ = ["ConstantPulseTemplate"]


class ConstantPulseTemplate(AtomicPulseTemplate):  # type: ignore
    def __init__(self, duration: ExpressionLike, amplitude_dict: Dict[ChannelID, ExpressionLike],
                 identifier: Optional[str] = None,
                 name: Optional[str] = None,
                 measurements: Optional[List[MeasurementDeclaration]] = None,
                 registry: PulseRegistryType=None) -> None:
        """An atomic pulse template qupulse representing a multi-channel pulse with constant values.
        
        Args:
            duration: Duration of the template
            amplitude_dict: Dictionary with values for the channels
            name: Name for the template. Not used by qupulse
        """
        super().__init__(identifier=identifier, measurements=measurements)

        # we special case numeric values in this PulseTemplate for performance reasons
        self._duration = duration if isinstance(duration, (float, int, TimeType)) else ExpressionScalar(duration)
        self._amplitude_dict: Mapping[ChannelID, Union[float, ExpressionScalar]] = {
            channel: value if isinstance(value, (float, int)) else ExpressionScalar(value)
            for channel, value in amplitude_dict.items()}

        if name is None:
            name = 'constant_pulse'
        self._name = name

        self._register(registry)

    def _as_expression(self):
        return self._amplitude_dict

    def __str__(self) -> str:
        return '<{} at %x{}: {}>'.format(self.__class__.__name__, '%x' % id(self), self._name)

    def get_serialization_data(self, serializer=None) -> Any:
        if serializer is not None:
            raise NotImplementedError("ConstantPulseTemplate does not implement legacy serialization.")
        data = super().get_serialization_data()
        data.update({
            'name': self._name,
            'duration': self._duration,
            'amplitude_dict': self._amplitude_dict,
            'measurements': self.measurement_declarations
        })
        return data

    @property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        """Returns an expression giving the integral over the pulse."""
        return {c: self.duration * self._amplitude_dict[c] for c in self._amplitude_dict}

    @cached_property
    def parameter_names(self) -> AbstractSet[str]:
        """The set of names of parameters required to instantiate this PulseTemplate."""
        parameters = []
        for amplitude in self._amplitude_dict.values():
            if hasattr(amplitude, 'variables'):
                parameters.extend(amplitude.variables)
        if hasattr(self._duration, 'variables'):
            parameters.extend(self._duration.variables)
        parameters.extend(self.measurement_parameters)
        return frozenset(parameters)

    @cached_property
    def duration(self) -> ExpressionScalar:
        """An expression for the duration of this PulseTemplate."""
        if isinstance(self._duration, ExpressionScalar):
            return self._duration
        else:
            return ExpressionScalar(self._duration)

    @property
    def defined_channels(self) -> AbstractSet['ChannelID']:
        """Returns the number of hardware output channels this PulseTemplate defines."""
        return self._amplitude_dict.keys()

    def build_waveform(self,
                       parameters: Dict[str, numbers.Real],
                       channel_mapping: Dict[ChannelID, Optional[ChannelID]]) -> Optional[Union[ConstantWaveform,
                                                                                                MultiChannelWaveform]]:
        logging.debug(f'build_waveform of ConstantPulse: channel_mapping {channel_mapping}, '
                      f'defined_channels {self.defined_channels}')

        # we very freely use duck-typing here to speed up cases where duration and amplitude values are already numeric
        duration = self._duration
        if hasattr(duration, 'evaluate_in_scope'):
            duration = duration.evaluate_in_scope(parameters)

        if duration > 0:
            constant_values = {}
            for channel, value in self._amplitude_dict.items():
                mapped_channel = channel_mapping[channel]
                if mapped_channel is not None:
                    if hasattr(value, 'evaluate_in_scope'):
                        value = value.evaluate_in_scope(parameters)
                    constant_values[mapped_channel] = value

            if constant_values:
                return ConstantWaveform.from_mapping(duration, constant_values)
        return None

    @property
    def initial_values(self) -> Dict[ChannelID, ExpressionScalar]:
        return {ch: ExpressionScalar(val) for ch, val in self._amplitude_dict.items()}

    @property
    def final_values(self) -> Dict[ChannelID, ExpressionScalar]:
        return {ch: ExpressionScalar(val) for ch, val in self._amplitude_dict.items()}
