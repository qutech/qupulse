"""This module contains the tools to setup a virtual AWG i.e. an AWG that forwards the program into a given callback.
This is handy to setup a simulation to test qupulse pulses."""
from typing import Tuple, Optional, Callable, Set

import numpy as np

from qupulse.utils.types import ChannelID, TimeType
from qupulse._program._loop import Loop, make_compatible, to_waveform
from qupulse.hardware.awgs.base import AWG


__all__ = ['VirtualAWG']


SamplingCallback = Callable[[np.ndarray], np.ndarray]
SamplingCallback.__doc__ = """Maps an array ov times to an array of voltages. The time array has to be ordered"""


def _create_sampling_callbacks(program: Loop, channels, voltage_transformations) -> Tuple[float,
                                                                                          Tuple[SamplingCallback, ...]]:
    waveform = to_waveform(program)

    duration = float(waveform.duration)

    def get_callback(channel: Optional[ChannelID], voltage_transformation):
        if channel is None:
            return None
        else:
            def sample_channel(time: np.ndarray):
                return voltage_transformation(waveform.get_sampled(channel, time))

            return sample_channel

    callbacks = [get_callback(channel, voltage_transformation)
                 for channel, voltage_transformation in zip(channels, voltage_transformations)]
    return duration, tuple(callbacks)


class VirtualAWG(AWG):
    """This class allows registering callbacks the given program is fed into.

    TODO:
     - adaptive sample rate (requires program analysis)"""

    def __init__(self, identifier: str, channels: int):
        super().__init__(identifier)

        self._programs = {}
        self._current_program = None
        self._channels = tuple(range(channels))

        self._function_handle_callback = None

    @property
    def num_channels(self) -> int:
        return len(self._channels)

    @property
    def num_markers(self) -> int:
        return 0

    def upload(self, name: str,
               program: Loop,
               channels: Tuple[Optional[ChannelID], ...],
               markers: Tuple[Optional[ChannelID], ...],
               voltage_transformation: Tuple[Optional[Callable], ...],
               force: bool=False):
        if name in self._programs and not force:
            raise RuntimeError('Program already known')

        self._programs[name] = (program, channels, voltage_transformation)

    def remove(self, name: str):
        self._programs.pop(name)

    def clear(self):
        self._programs.clear()
        self._current_program = None

    def arm(self, name: Optional[str]):
        self._current_program = name

    @property
    def programs(self) -> Set[str]:
        return set(self._programs.keys())

    @property
    def sample_rate(self) -> float:
        return float('nan')

    def set_function_handle_callback(self,
                                     callback: Optional[Callable[[float, Tuple[SamplingCallback, ...]], None]]):
        """When run current program is called the given callback is called with the first positional argument being the
        duration and following arguments being sampling callbacks as defined above."""
        self._function_handle_callback = callback

    def run_current_program(self):
        (program, channels, voltage_transformations) = self._programs[self._current_program]

        if self._function_handle_callback is not None:
            duration, sample_callbacks = _create_sampling_callbacks(program, channels, voltage_transformations)
            self._function_handle_callback(duration, sample_callbacks)
