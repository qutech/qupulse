"""This module contains the tools to setup a virtual AWG i.e. an AWG that forwards the program into a given callback.
This is handy to setup a simulation to test qupulse pulses."""
from typing import Union, Sequence, Tuple, Optional, Callable, Set

import numpy as np

from qupulse.utils.types import ChannelID, TimeType
from qupulse._program._loop import Loop, make_compatible
from qupulse.hardware.awgs.base import AWG


__all__ = ['VirtualAWG']


def _feed_into_callable(program: Loop, callback, channels, dt):
    """Maybe there is already a function somewhere for this?"""
    binary_waveforms = {}
    reverse_lookup = {}
    time_array = np.arange(1024, dtype=float) * float(dt)
    for p in program.get_depth_first_iterator():
        if p.waveform and p.waveform not in binary_waveforms:
            # this waveform is not sampled yet
            n_samples = p.waveform.duration / dt

            if time_array.size < n_samples:
                # we need to extend the time array
                time_array = np.arange(n_samples, dtype=float) * float(dt)

            sample_times = time_array[:n_samples]
            sampled = np.nan((len(channels), n_samples))
            for idx, channel in enumerate(channels):
                p.waveform.get_sampled(channel, sample_times, output_array=sampled[idx, :])

            result_hash = hash(sampled.tobytes())
            if result_hash in reverse_lookup:
                # save memory for binary equivalent waveforms
                np.testing.assert_array_equal(sampled, reverse_lookup[result_hash])
                binary_waveforms[p.waveform] = reverse_lookup[result_hash]
            else:
                binary_waveforms[p.waveform] = sampled
                reverse_lookup[result_hash] = sampled

    stack = [program]
    while stack:
        current = stack.pop()
        if current.waveform:
            sampled = binary_waveforms[current.waveform]
            for _ in range(current.repetition_count):
                callback(sampled)

        else:
            for _ in range(current.repetition_count):
                stack.extend(reversed(current))


class VirtualAWG(AWG):
    """This class allows registering callbacks the given program is fed into."""

    def __init__(self, identifier: str, channels: int):
        super().__init__(identifier)

        self._programs = {}
        self._current_program = None
        self._channels = tuple(range(channels))

        self._fixed_sample_rate_callbacks = {}

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

    def set_fixed_sample_rate_callback(self, name: str, callback: Callable[[np.ndarray], None], sample_rate):
        if callback is None:
            self._fixed_sample_rate_callbacks.pop(name, None)
        else:
            self._fixed_sample_rate_callbacks[name] = (callback, TimeType(sample_rate))

    def run_current_program(self):
        (program, channels, voltage_transformation) = self._programs[self._current_program]

        for callback, sample_rate in self._fixed_sample_rate_callbacks.values():
            dt = 1/sample_rate
            c_program = program.copy()

            # assert all waveforms have a length that is a multiple of the time per sample
            make_compatible(c_program, 0, dt, sample_rate)




