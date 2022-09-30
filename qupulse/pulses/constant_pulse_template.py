"""This module defines the ConstantPulseTemplate, a pulse tempalte representating a pulse with constant values on all channels

Classes:
    - ConstantPulseTemplate: Defines a pulse via channel-value pairs
    - ConstantPulseWaveform: A waveform instantiated from a TablePulseTemplate by providing values for its
        declared parameters.
"""

import logging
import numbers
from typing import Any, Dict, List, Optional, Set, Union

from qupulse._program.waveforms import ConstantWaveform
from qupulse.expressions import ExpressionScalar
from qupulse.parameter_scope import Scope
from qupulse.pulses.multi_channel_pulse_template import MultiChannelWaveform
from qupulse.pulses.parameters import ParameterNotProvidedException
from qupulse.pulses.pulse_template import (AtomicPulseTemplate, ChannelID,
                                           Loop, MeasurementDeclaration,
                                           PulseTemplate, Transformation,
                                           TransformingWaveform)

__all__ = ["ConstantPulseTemplate"]


class ConstantPulseTemplate(AtomicPulseTemplate):  # type: ignore

    def __init__(self, duration: float, amplitude_dict: Dict[str, Any], identifier: Optional[str] = None,
                 name: Optional[str] = None, measurements: Optional[List[MeasurementDeclaration]] = [], **kwargs: Any) -> None:
        """ A qupulse waveform representing a multi-channel pulse with constant values
        
        Args:
            duration: Duration of the template
            amplitude_dict: Dictionary with values for the channels
            name: Name for the template
        
        """
        super().__init__(identifier=identifier, measurements=measurements, **kwargs)

        self._duration = ExpressionScalar(duration)
        self._amplitude_dict = amplitude_dict

        if name is None:
            name = 'constant_pulse'
        self._name = name

    def _as_expression(self):
        return self._amplitude_dict

    def __str__(self) -> str:
        return '<{} at %x{}: {}>'.format(self.__class__.__name__, '%x' % id(self), self._name)

    def build_sequence(self) -> None:
        return

    def get_serialization_data(self) -> Any:
        data = super().get_serialization_data()
        data.update({'name': self._name, 'duration': self._duration, '#amplitudes': self._amplitude_dict})
        return data

    @property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        """Returns an expression giving the integral over the pulse."""
        return {c: self.duration * self._amplitude_dict[c] for c in self._amplitude_dict}

    @property
    def parameter_names(self) -> Set[str]:
        """The set of names of parameters required to instantiate this PulseTemplate."""
        return set()

    @property
    def is_interruptable(self) -> bool:
        """Return true, if this PulseTemplate contains points at which it can halt if interrupted.
        """
        return False

    @property
    def duration(self) -> ExpressionScalar:
        """An expression for the duration of this PulseTemplate."""
        return (self._duration)

    @property
    def defined_channels(self) -> Set['ChannelID']:
        """Returns the number of hardware output channels this PulseTemplate defines."""
        return set(self._amplitude_dict.keys())

    def requires_stop(self) -> bool:  # from SequencingElement
        return False

    def _internal_create_program(self, *,
                                 scope: Scope,
                                 measurement_mapping: Dict[str, Optional[str]],
                                 channel_mapping: Dict[ChannelID, Optional[ChannelID]],
                                 global_transformation: Optional[Transformation],
                                 to_single_waveform: Set[Union[str, PulseTemplate]],
                                 parent_loop: Loop) -> None:
        """ removed from qupulse docstring """
        try:
            parameters = {parameter_name: scope[parameter_name].get_value()
                          for parameter_name in self.parameter_names}
        except KeyError as e:
            raise ParameterNotProvidedException(str(e)) from e

        waveform = self.build_waveform(parameters=parameters,
                                       channel_mapping=channel_mapping)
        if waveform:
            measurements: List[Any] = []
            measurements = self.get_measurement_windows(parameters=parameters,
                                                        measurement_mapping=measurement_mapping)

            if global_transformation:
                waveform = TransformingWaveform(waveform, global_transformation)

            parent_loop.add_measurements(measurements=measurements)
            parent_loop.append_child(waveform=waveform)

    def build_waveform(self,
                       parameters: Dict[str, numbers.Real],
                       channel_mapping: Dict[ChannelID, Optional[ChannelID]]) -> Optional[Union[ConstantWaveform,
                                                                                                MultiChannelWaveform]]:
        logging.debug(f'build_waveform of ConstantPulse: channel_mapping {channel_mapping}, defined_channels {self.defined_channels} ')
        if all(channel_mapping.get(channel, None) is None
               for channel in self.defined_channels):
            return None

        
        waveforms = [ConstantWaveform(self.duration, amplitude, channel)
                     for channel, amplitude in self._amplitude_dict.items() if channel_mapping[channel] is not None]

        if len(waveforms) == 1:
            return waveforms.pop()
        else:
            return MultiChannelWaveform(waveforms)
