from typing import NamedTuple, Any, Set, Callable, Dict, Tuple

from ctypes import c_int64 as MutableInt

from qctoolkit.hardware.awgs import AWG
from qctoolkit.hardware.dacs import DAC
from qctoolkit.hardware.program import MultiChannelProgram, Loop

from qctoolkit import ChannelID

import numpy as np


__all__ = ['PlaybackChannel', 'HardwareSetup']


PlaybackChannel = NamedTuple('PlaybackChannel', [('awg', AWG),
                                                 ('channel_on_awg', Any),
                                                 ('voltage_transformation', Callable[[np.ndarray], np.ndarray])])
PlaybackChannel.__new__.__defaults__ = (lambda v: v,)
PlaybackChannel.__doc__ += ': Properties of an actual hardware channel'
PlaybackChannel.awg.__doc__ = 'The AWG the channel is defined on'
PlaybackChannel.channel_on_awg.__doc__ = 'The channel\'s index(starting with 0) on the AWG.'
PlaybackChannel.voltage_transformation.__doc__ = \
    'A transformation that is applied to the pulses on the channel.\
    One use case is to scale up the voltage if an amplifier is inserted.'
PlaybackChannel.__hash__ = lambda pbc: hash((id(pbc.awg), pbc.channel_on_awg))


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
        self.__dacs = []

        self.__channel_map = dict()  # type: Dict[ChannelID, Set[PlaybackChannel]]

        self.__registered_programs = dict()  # type: Dict[str, RegisteredProgram]

    def register_program(self, name: str, instruction_block, run_callback=lambda: None, update=False):
        if not callable(run_callback):
            raise TypeError('The provided run_callback is not callable')

        mcp = MultiChannelProgram(instruction_block)

        measurement_windows = dict()

        for program in mcp.programs.values():
            program.get_measurement_windows(measurement_windows=measurement_windows)

        for mw_name, begin_length_list in measurement_windows.items():
            measurement_windows[mw_name] = sorted(set(begin_length_list))

        handled_awgs = set()
        for channels, program in mcp.programs:
            awgs_to_upload_to = dict()
            for channel_id in channels:
                if channel_id in channels:
                    pbc = self.__channel_map[channel_id]
                    awgs_to_upload_to.get(pbc.awg, [None]*pbc.awg.num_channels)[pbc.channel_on_awg] = channel_id

            for awg, channel_ids in awgs_to_upload_to.items():
                if awg in handled_awgs:
                    raise ValueError('AWG has two programs')
                else:
                    handled_awgs.add(awg)
                awg.upload(name, program=program, channels=channel_ids, force=update)

        for dac in self.__dacs:
            dac.register_measurement_windows(name, measurement_windows)

        self.__registered_programs[name] = RegisteredProgram(program=mcp,
                                                             measurement_windows=measurement_windows,
                                                             run_callback=run_callback,
                                                             awgs_to_upload_to=
                                                             set(awg for awg, _ in awgs_to_upload_to.items()))

    def arm_program(self, name):
        """Assert program is in memory. Hardware will wait for trigger event"""
        for awg in self.__registered_programs[name].awgs_to_upload_to:
            awg.arm(name)
        for dac in self.__dacs:
            dac.arm_program(name)

    def run_program(self, name):
        """Calls arm program and starts it using the run callback"""
        self.arm_program(name)
        self.__registered_programs[name].run_callback()

    def set_channel(self, identifier: ChannelID, playback_channel: PlaybackChannel):
        for ch_id, pbc_set in self.__channel_map.items():
            if playback_channel in pbc_set:
                raise ValueError('Channel already registered as playback channel for channel {}'.format(ch_id))
        self.__channel_map[identifier] = playback_channel

    def rm_channel(self, identifier: ChannelID):
        self.__channel_map.pop(identifier)

    def registered_playback_channels(self) -> Set[PlaybackChannel]:
        return set(pbc for pbc_set in self.__channel_map.values() for pbc in pbc_set)








