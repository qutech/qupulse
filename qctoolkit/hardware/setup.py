from typing import NamedTuple, Any, Set, Callable, Dict, Tuple, Union
import itertools
from collections import defaultdict, deque

from ctypes import c_int64 as MutableInt

from qctoolkit.hardware.awgs import AWG
from qctoolkit.hardware.dacs import DAC
from qctoolkit.hardware.program import MultiChannelProgram, Loop

from qctoolkit import ChannelID

import numpy as np


__all__ = ['PlaybackChannel', 'MarkerChannel', 'HardwareSetup']


class _SingleChannel:
    """An actual hardware channel"""
    def __init__(self, awg: AWG, channel_on_awg: int):
        self.awg = awg
        """The AWG the channel is defined on"""

        self.channel_on_awg = channel_on_awg
        """The channel's index(starting with 0) on the AWG."""

    def __hash__(self):
        return hash((id(self.awg), self.channel_on_awg, type(self)))

    def __eq__(self, other):
        return hash(self) == hash(other)


class PlaybackChannel(_SingleChannel):
    """A hardware channel that is not a marker"""
    def __init__(self, awg: AWG, channel_on_awg: int,
                 voltage_transformation: Callable[[np.ndarray], np.ndarray]=lambda x: x):
        if channel_on_awg >= awg.num_channels:
            raise ValueError('Can not create PlayBack channel {}. AWG only has {} channels'.format(channel_on_awg,
                                                                                                   awg.num_channels))
        super().__init__(awg=awg, channel_on_awg=channel_on_awg)

        self.voltage_transformation = voltage_transformation
        """A transformation that is applied to the pulses on the channel. One use case is to scale up the voltage if an
        amplifier is inserted."""


class MarkerChannel(_SingleChannel):
    """A hardware channel that can only take two values"""
    def __init__(self, awg: AWG, channel_on_awg: int):
        if channel_on_awg >= awg.num_markers:
            raise ValueError('Can not create MarkerBack channel {}. AWG only has {} channels'.format(channel_on_awg,
                                                                                                     awg.num_markers))
        super().__init__(awg=awg, channel_on_awg=channel_on_awg)


RegisteredProgram = NamedTuple('RegisteredProgram', [('program', MultiChannelProgram),
                                                     ('measurement_windows', Dict[str, Tuple[float, float]]),
                                                     ('run_callback', Callable),
                                                     ('awgs_to_upload_to', Set[AWG])])


class HardwareSetup:
    """Representation of the hardware setup.

    The class takes an instruction block, forms it into possibly channel dependent programs
    and registers the programs at the AWGs which modify their program to fit to their capabilities. The class also
    extracts the measurement windows(with absolute times) and hands them over to the DACs which will do further
    processing."""
    def __init__(self):
        self._dacs = []

        self._channel_map = dict()  # type: Dict[ChannelID, Set[SingleChannel]]

        self._registered_programs = dict()  # type: Dict[str, RegisteredProgram]

    def register_program(self, name: str, instruction_block, run_callback=lambda: None, update=False) -> None:
        if not callable(run_callback):
            raise TypeError('The provided run_callback is not callable')

        mcp = MultiChannelProgram(instruction_block)

        temp_measurement_windows = defaultdict(deque)
        for program in mcp.programs.values():
            for mw_name, begins_lengths in program.get_measurement_windows().items():
                temp_measurement_windows[mw_name].append(begins_lengths)

        measurement_windows = dict()
        while temp_measurement_windows:
            mw_name, begins_lengths_deque = temp_measurement_windows.popitem()
            measurement_windows[mw_name] = (
                np.concatenate(tuple(begins for begins, _ in begins_lengths_deque)),
                np.concatenate(tuple(lengths for _, lengths in begins_lengths_deque))
            )

        handled_awgs = set()
        for channels, program in mcp.programs.items():
            awgs_to_channel_info = dict()

            def get_default_info(awg):
                return ([None] * awg.num_channels,
                        [None] * awg.num_channels,
                        [None] * awg.num_markers)

            for channel_id in channels:
                for single_channel in self._channel_map[channel_id]:
                    playback_ids, voltage_trafos, marker_ids = \
                        awgs_to_channel_info.setdefault(single_channel.awg, get_default_info(single_channel.awg))

                    if isinstance(single_channel, PlaybackChannel):
                        playback_ids[single_channel.channel_on_awg] = channel_id
                        voltage_trafos[single_channel.channel_on_awg] = single_channel.voltage_transformation
                    elif isinstance(single_channel, MarkerChannel):
                        marker_ids[single_channel.channel_on_awg] = channel_id

            for awg, (playback_ids, voltage_trafos, marker_ids) in awgs_to_channel_info.items():
                if awg in handled_awgs:
                    raise ValueError('AWG has two programs')
                else:
                    handled_awgs.add(awg)
                awg.upload(name,
                           program=program,
                           channels=tuple(playback_ids),
                           markers=tuple(marker_ids),
                           force=update,
                           voltage_transformation=tuple(voltage_trafos))

        for dac in self._dacs:
            dac.register_measurement_windows(name, measurement_windows)

        self._registered_programs[name] = RegisteredProgram(program=mcp,
                                                            measurement_windows=measurement_windows,
                                                            run_callback=run_callback,
                                                            awgs_to_upload_to=handled_awgs)

    def arm_program(self, name) -> None:
        """Assert program is in memory. Hardware will wait for trigger event"""
        if name not in self._registered_programs:
            raise KeyError('{} is not a registered program'.format(name))
        for awg in self._registered_programs[name].awgs_to_upload_to:
            awg.arm(name)
        for dac in self._dacs:
            dac.arm_program(name)

    def run_program(self, name) -> None:
        """Calls arm program and starts it using the run callback"""
        self.arm_program(name)
        self._registered_programs[name].run_callback()

    def set_channel(self, identifier: ChannelID, single_channel: Union[PlaybackChannel, MarkerChannel]) -> None:
        for ch_id, channel_set in self._channel_map.items():
            if single_channel in channel_set:
                raise ValueError('Channel already registered as {} for channel {}'.format(
                    type(self._channel_map[ch_id]).__name__, ch_id))

        if isinstance(single_channel, (PlaybackChannel, MarkerChannel)):
            self._channel_map.setdefault(identifier, set()).add(single_channel)
        else:
            raise ValueError('Channel must be either a playback or a marker channel')

    def rm_channel(self, identifier: ChannelID) -> None:
        self._playback_channel_map.pop(identifier)

    def registered_channels(self) -> Set[PlaybackChannel]:
        return self._channel_map.copy()

    def register_dac(self, dac):
        if dac in self._dacs:
            raise ValueError('DAC already known {}'.format(str(dac)))
        self._dacs.append(dac)

    @property
    def registered_programs(self) -> Dict:
        return self._registered_programs








