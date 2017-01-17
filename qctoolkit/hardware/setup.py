from typing import NamedTuple, Any, Callable

from ctypes import c_int64 as MutableInt

from qctoolkit.hardware.awgs import AWG
from qctoolkit.hardware.dacs import DAC
from qctoolkit.hardware.program import MultiChannelProgram, Loop

import numpy as np


__all__ = ['PlaybackChannel', 'HardwareSetup']


PlaybackChannel = NamedTuple('PlaybackChannel', [('awg', AWG),
                                                 ('channel_on_awg', Any),
                                                 ('voltage_transformation', Callable[[np.ndarray], np.ndarray])])
PlaybackChannel.__new__.__defaults__ = (lambda v: v,)
PlaybackChannel.__doc__ += ': Properties of an actual hardware channel'
PlaybackChannel.awg.__doc__ = 'The AWG the channel is defined on'
PlaybackChannel.channel_on_awg.__doc__ = 'The channel\'s ID on the AWG.'
PlaybackChannel.voltage_transformation.__doc__ = \
    'A transformation that is applied to the pulses on the channel.\
    One use case is to scale up the voltage if an amplifier is inserted.'



class HardwareSetup:
    """Representation of the hardware setup.

    The class takes an instruction block, forms it into possibly channel dependent programs
    and registers the programs at the AWGs which modify their program to fit to their capabilities. The class also
    extracts the measurement windows(with absolute times) and hands them over to the DACs which will do further
    processing."""
    def __init__(self):
        self.__dacs = []

        self.__channel_map = dict()  # type: Dict[ChannelID, PlaybackChannel]
        self.__awgs = []  # type: List[AWG, List[ChannelID, Any]]

        self.__registered_programs = dict()

    def register_program(self, name: str, instruction_block, run_callback=None):
        mcp = MultiChannelProgram(instruction_block)

        measurement_windows = dict()

        def extract_measurement_windows(loop: Loop, offset: MutableInt):
            if loop.is_leaf():
                for (mw_name, begin, length) in loop.instruction.measurement_windows:
                    measurement_windows.get(mw_name, default=[]).append(begin + offset.value, length)
                offset.value += loop.instruction.waveform.duration
            else:
                for sub_loop in loop:
                    extract_measurement_windows(sub_loop, offset)
        for program in mcp.programs.values():
            extract_measurement_windows(program, MutableInt(0))

        for channels, program in mcp.programs:
            pass

        try:
            for dac in self.__dacs:
                dac.register_measurement_windows(name, measurement_windows)
        except:
            raise

        self.__registered_programs[name] = (mcp, measurement_windows,
                                            lambda: None if run_callback is None else run_callback)

        raise NotImplementedError()

    def arm_program(self, name):
        """Assert program is in memory. Hardware will wait for trigger event"""
        raise NotImplementedError()

    def run_program(self, name):
        """Calls arm program and starts it using the run callback"""
        raise NotImplementedError()

    def set_channel(self, identifier: ChannelID, channel: PlaybackChannel):
        self.__channel_map[identifier] = channel







