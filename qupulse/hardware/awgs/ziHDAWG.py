import sys
import functools
from typing import List, Tuple, Set, NamedTuple, Callable, Optional, Any, Sequence, cast, Generator, Union, Dict
from enum import Enum

# Zurich Instruments LabOne python API distributed via the Python Package Index.
import zhinst.ziPython
import zhinst.utils
import numpy as np
import textwrap
import time

from qupulse.utils.types import ChannelID
from qupulse._program._loop import Loop, make_compatible
from qupulse.hardware.awgs.base import AWG, ChannelNotFoundException

assert (sys.byteorder == 'little')


def valid_channel(function_object):
    """Check if channel is a valid AWG channels. Expects channel to be 2nd argument after self."""
    @functools.wraps(function_object)
    def valid_fn(*args, **kwargs):
        if len(args) < 1:
            raise TypeError('Channel is an required argument.')
        channel = args[1]  # Expect channel to be second positional argument after self.
        if channel not in range(1, 8):
            raise ChannelNotFoundException(channel)
        value = function_object(*args, **kwargs)
        return value
    return valid_fn


class HDAWGRepresentation:
    """HDAWGRepresentation represents an HDAWG8 instruments and manages a LabOne data server api session. A data server
    must be running and the device be discoverable. Channels are per default grouped into pairs."""

    def __init__(self, device_serial=None,
                 device_interface='1GbE',
                 data_server_addr='localhost',
                 data_server_port=8004,
                 api_level_number=6,
                 external_trigger=False, reset=False):
        """
        :param device_serial:     Device serial that uniquely identifies this device to the LabOne data server
        :param device_interface:  Either '1GbE' for ethernet or 'USB'
        :param data_server_addr:  Data server address. Must be already running. Default: localhost
        :param data_server_port:  Data server port. Default: 8004 for HDAWG, MF and UHF devices
        :param api_level_number:  Version of API to use for the session, higher number, newer. Default: 6 most recent
        :param external_trigger:  Not supported yet
        :param reset:             Reset device before initialization
        """
        self._api_session = zhinst.ziPython.ziDAQServer(data_server_addr, data_server_port, api_level_number)
        zhinst.utils.api_server_version_check(self.api_session)  # Check equal data server and api version.
        self.api_session.connectDevice(device_serial, device_interface)
        self._dev_ser = device_serial

        if external_trigger:
            raise NotImplementedError()  # pragma: no cover

        if reset:
            # TODO: Check if utils function is sufficient, or a custom reset function is required.
            # Create a base configuration: Disable all available outputs, awgs, demods, scopes,...
            zhinst.utils.disable_everything(self.api_session, self.serial)

        self.initialize()

        self._channel_pair_AB = HDAWGChannelPair(self, (1, 2), str(self.serial) + '_AB')
        self._channel_pair_CD = HDAWGChannelPair(self, (3, 4), str(self.serial) + '_CD')
        self._channel_pair_EF = HDAWGChannelPair(self, (5, 6), str(self.serial) + '_EF')
        self._channel_pair_GH = HDAWGChannelPair(self, (7, 8), str(self.serial) + '_GH')

    @property
    def channel_pair_AB(self) -> 'HDAWGChannelPair':
        return self._channel_pair_AB

    @property
    def channel_pair_CD(self) -> 'HDAWGChannelPair':
        return self._channel_pair_CD

    @property
    def channel_pair_EF(self) -> 'HDAWGChannelPair':
        return self._channel_pair_EF

    @property
    def channel_pair_GH(self) -> 'HDAWGChannelPair':
        return self._channel_pair_GH

    @property
    def api_session(self) -> zhinst.ziPython.ziDAQServer:
        return self._api_session

    @property
    def serial(self) -> str:
        return self._dev_ser

    def initialize(self) -> None:
        settings = []
        settings.append(['/{}/system/awg/channelgrouping'.format(self.serial),
                         HDAWGChannelGrouping.CHAN_GROUP_4x2.value])
        settings.append(['/{}/awgs/*/time'.format(self.serial), HDAWGSamplingRate.AWG_RATE_2400MHZ.value])
        settings.append(['/{}/sigouts/*/range'.format(self.serial), HDAWGVoltageRange.RNG_1V.value])
        settings.append(['/{}/awgs/*/outputs/*/amplitude'.format(self.serial), 1.0])  # Default amplitude factor 1.0
        settings.append(['/{}/awgs/*/outputs/*/modulation/mode'.format(self.serial), HDAWGModulationMode.OFF.value])
        settings.append(['/{}/awgs/*/userregs/*'.format(self.serial), 0])  # Reset all user registers to 0.

        self.api_session.set(settings)
        self.api_session.sync()  # Global sync: Ensure settings have taken effect on the device.

    def reset(self) -> None:
        # TODO: Check if utils function is sufficient to reset device.
        zhinst.utils.disable_everything(self.api_session, self.serial)
        self.initialize()
        self.channel_pair_AB.clear()
        self.channel_pair_CD.clear()
        self.channel_pair_EF.clear()
        self.channel_pair_GH.clear()

    @valid_channel
    def offset(self, channel: int, voltage: float = None) -> float:
        """Query channel offset voltage and optionally set it."""
        node_path = '/{}/sigouts/{}/offset'.format(self.serial, channel-1)
        if voltage is not None:
            self.api_session.setDouble(node_path, voltage)
            self.api_session.sync()  # Global sync: Ensure settings have taken effect on the device.
        return self.api_session.getDouble(node_path)

    @valid_channel
    def range(self, channel: int, voltage: float = None) -> float:
        """Query channel voltage range and optionally set it. The instruments selects the next higher available range.
        This is the one-sided range Vp. Total range: -Vp...Vp"""
        node_path = '/{}/sigouts/{}/range'.format(self.serial, channel-1)
        if voltage is not None:
            self.api_session.setDouble(node_path, voltage)
            self.api_session.sync()  # Global sync: Ensure settings have taken effect on the device.
        return self.api_session.getDouble(node_path)

    @valid_channel
    def output(self, channel: int, status: bool = None) -> bool:
        """Query channel signal output status (enabled/disabled) and optionally set it. Corresponds to front LED."""
        node_path = '/{}/sigouts/{}/on'.format(self.serial, channel-1)
        if status is not None:
            self.api_session.setDouble(node_path, int(status))
            self.api_session.sync()  # Global sync: Ensure settings have taken effect on the device.
        return bool(self.api_session.getDouble(node_path))


class HDAWGChannelGrouping(Enum):
    """How many independent sequencers should run on the AWG and how the outputs should be grouped by sequencer."""
    CHAN_GROUP_4x2 = 0  # 4x2 with HDAWG8; 2x2 with HDAWG4.  /dev.../awgs/0..3/
    CHAN_GROUP_2x4 = 1  # 2x4 with HDAWG8; 1x4 with HDAWG4.  /dev.../awgs/0 & 2/
    CHAN_GROUP_1x8 = 2  # 1x8 with HDAWG8.                   /dev.../awgs/0/


class HDAWGVoltageRange(Enum):
    """All available voltage ranges for the HDAWG wave outputs. Define maximum output voltage."""
    RNG_5V = 5
    RNG_4V = 4
    RNG_3V = 3
    RNG_2V = 2
    RNG_1V = 1
    RNG_800mV = 0.8
    RNG_600mV = 0.6
    RNG_400mV = 0.4
    RNG_200mV = 0.2


class HDAWGModulationMode(Enum):
    """Modulation mode of waveform generator."""
    OFF = 0  # AWG output goes directly to signal output.
    SINE_1 = 1  # AWG output multiplied with sine generator signal 0.
    SINE_2 = 2  # AWG output multiplied with sine generator signal 1.
    FG_1 = 3  # AWG output multiplied with function generator signal 0. Requires FG option.
    FG_2 = 4  # AWG output multiplied with function generator signal 1. Requires FG option.
    ADVANCED = 5  # AWG output modulates corresponding sines from modulation carriers.


class HDAWGSamplingRate(Enum):
    """Supported sampling rates of the AWG."""
    AWG_RATE_2400MHZ = 0  # Constant to set sampling rate to 2.4 GHz.
    AWG_RATE_1200MHZ = 1  # Constant to set sampling rate to 1.2 GHz.
    AWG_RATE_600MHZ = 2  # Constant to set sampling rate to 600 MHz.
    AWG_RATE_300MHZ = 3  # Constant to set sampling rate to 300 MHz.
    AWG_RATE_150MHZ = 4  # Constant to set sampling rate to 150 MHz.
    AWG_RATE_75MHZ = 5  # Constant to set sampling rate to 75 MHz.
    AWG_RATE_37P5MHZ = 6  # Constant to set sampling rate to 37.5 MHz.
    AWG_RATE_18P75MHZ = 7  # Constant to set sampling rate to 18.75MHz.
    AWG_RATE_9P4MHZ = 8  # Constant to set sampling rate to 9.4 MHz.
    AWG_RATE_4P5MHZ = 9  # Constant to set sampling rate to 4.5 MHz.
    AWG_RATE_2P34MHZ = 10  # Constant to set sampling rate to 2.34MHz.
    AWG_RATE_1P2MHZ = 11  # Constant to set sampling rate to 1.2 MHz.
    AWG_RATE_586KHZ = 12  # Constant to set sampling rate to 586 kHz.
    AWG_RATE_293KHZ = 13  # Constant to set sampling rate to 293 kHz.

    def exact_rate(self):
        """Calculate exact sampling rate based on (2.4 GSa/s)/2^n, where n is the current enum value."""
        return 2.4e9 / 2 ** self.value


class HDAWGChannelPair(AWG):
    """Represents a channel pair of the Zurich Instruments HDAWG as an independent AWG entity.
    It represents a set of channels that have to have(hardware enforced) the same:
        -control flow
        -sample rate

    It keeps track of the AWG state and manages waveforms and programs on the hardware.
    """

    def __init__(self, hdawg_device: HDAWGRepresentation, channels: Tuple[int, int], identifier: str):
        super().__init__(identifier)
        self._device = hdawg_device

        if channels not in ((1, 2), (3, 4), (5, 6), (7, 8)):
            raise ValueError('Invalid channel pair: {}'.format(channels))
        self._channels = channels

        self._awg_module = self.device.api_session.awgModule()
        self.awg_module.set('awgModule/device', self.device.serial)
        self.awg_module.set('awgModule/index', self.awg_group_index)
        self.awg_module.execute()
        # Seems creating AWG module sets SINGLE (single execution mode of sequence) to 0.
        self.device.api_session.setInt('/{}/awgs/{}/single'.format(self.device.serial, self.awg_group_index), 1)

        self._known_programs = dict()  # type: Dict[str, TaborProgramMemory]

    @property
    def num_channels(self) -> int:
        """Number of channels"""
        return 2

    @property
    def num_markers(self) -> int:
        """Number of marker channels"""
        return 2  # Actually 4 available.

    def upload(self, name: str,
               program: Loop,
               channels: Tuple[Optional[ChannelID], Optional[ChannelID]],
               markers: Tuple[Optional[ChannelID], Optional[ChannelID]],
               voltage_transformation: Tuple[Callable, Callable],
               force: bool = False) -> None:
        """Upload a program to the AWG.

        Physically uploads all waveforms required by the program - excluding those already present -
        to the device and sets up playback sequences accordingly.
        This method should be cheap for program already on the device and can therefore be used
        for syncing. Programs that are uploaded should be fast(~1 sec) to arm.

        Args:
            name: A name for the program on the AWG.
            program: The program (a sequence of instructions) to upload.
            channels: Tuple of length num_channels that ChannelIDs of  in the program to use. Position in the list
            corresponds to the AWG channel
            markers: List of channels in the program to use. Position in the List in the list corresponds to
            the AWG channel
            voltage_transformation: transformations applied to the waveforms extracted rom the program. Position
            in the list corresponds to the AWG channel
            force: If a different sequence is already present with the same name, it is
                overwritten if force is set to True. (default = False)
        """
        if len(channels) != self.num_channels:
            raise ValueError('Channel ID not specified')
        if len(markers) != self.num_markers:
            raise ValueError('Markers not specified')
        if len(voltage_transformation) != self.num_channels:
            raise ValueError('Wrong number of voltage transformations')

        # Adjust program to fit criteria.
        make_compatible(program,
                        minimal_waveform_length=16,
                        waveform_quantum=16,
                        sample_rate=self.sample_rate)

        awg_program = textwrap.dedent("""\
        const N = 4096;
        wave gauss_pos = 1.0*gauss(N, N/2, N/8);
        wave gauss_neg = -1.0*gauss(N, N/2, N/8);
        while (true) {  
            playWave(gauss_pos);
            waitWave();
            playWave(2, gauss_neg);
            waitWave();
        }
        """)

        # Transfer the AWG sequence program. Compilation starts automatically in case of sourcestring.
        self.awg_module.set('awgModule/compiler/sourcestring', awg_program)

        # Compilation started
        while self.awg_module.getInt('awgModule/compiler/status') == -1:
            time.sleep(0.1)

        if self.awg_module.getInt('awgModule/compiler/status') == 1:
            raise HDAWGCompilationException(self.awg_module.getString('awgModule/compiler/statusstring'))

        if self.awg_module.getInt('awgModule/compiler/status') == 0:
            pass  # Compilation successful
        if self.awg_module.getInt('awgModule/compiler/status') == 2:
            print("Compiler warning: ", self.awg_module.getString('awgModule/compiler/statusstring'))

        while ((self.awg_module.getDouble('awgModule/progress') < 1.0) and
               (self.awg_module.getInt('awgModule/elf/status') != 1)):
            time.sleep(0.2)

        if self.awg_module.getInt('awgModule/elf/status') == 0:
            pass  # Upload to the instrument successful.
        if self.awg_module.getInt('awgModule/elf/status') == 1:
            raise HDAWGUploadException()

    def remove(self, name: str) -> None:
        """Remove a program from the AWG.

        Also discards all waveforms referenced only by the program identified by name.

        Args:
            name: The name of the program to remove.
        """
        raise NotImplementedError()  # pragma: no cover

    def clear(self) -> None:
        """Removes all programs and waveforms from the AWG.

        Caution: This affects all programs and waveforms on the AWG, not only those uploaded using qupulse!
        """
        raise NotImplementedError()  # pragma: no cover

    def arm(self, name: str) -> None:
        """Load the program 'name' and arm the device for running it. If name is None the awg will "dearm" its current
        program."""
        raise NotImplementedError()  # pragma: no cover

    @property
    def programs(self) -> Set[str]:
        """The set of program names that can currently be executed on the hardware AWG."""
        return set(program.name for program in self._known_programs.keys())

    @property
    def sample_rate(self) -> float:
        """The default sample rate of the AWG."""
        node_path = '/{}/awgs/{}/time'.format(self.device.serial, self.awg_group_index)
        sample_rate_num = self.device.api_session.getInt(node_path)
        return HDAWGSamplingRate(sample_rate_num).exact_rate()

    @property
    def awg_group_index(self) -> int:
        """AWG node branch index assuming 4x2 channel grouping. Then 0...3 will give appropriate index of group."""
        return int(np.ceil(self._channels[0]/2.0)-1)

    @property
    def device(self) -> HDAWGRepresentation:
        """Reference to HDAWG represenation."""
        return self._device

    @property
    def awg_module(self) -> zhinst.ziPython.AwgModule:
        """Each AWG entity has its own awg module to manage program compilation and upload."""
        return self._awg_module

    def enable(self, status: bool = None) -> bool:
        """Start the AWG sequencer."""
        # There is also 'awgModule/awg/enable', which seems to have the same functionality.
        node_path = '/{}/awgs/{}/enable'.format(self.device.serial, self.awg_group_index)
        if status is not None:
            self.device.api_session.setInt(node_path, int(status))
            self.device.api_session.sync()  # Global sync: Ensure settings have taken effect on the device.
        return bool(self.device.api_session.getInt(node_path))

    def user_register(self, reg: int, value: int = None) -> int:
        """Query user registers (1-16) and optionally set it."""
        if reg not in range(1, 16):
            raise ValueError('{} not a valid (1-16) register.'.format(reg))
        node_path = '/{}/awgs/{}/userregs/{}'.format(self.device.serial, self.awg_group_index, reg)
        if value is not None:
            self.device.api_session.setInt(node_path, value)
            self.device.api_session.sync()
        return self.device.api_session.getInt(node_path)


class HDAWGException(Exception):
    """Base exception class for HDAWG errors."""
    pass


class HDAWGCompilationException(HDAWGException):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self) -> str:
        return "Compilation failed: {}".format(self.msg)


class HDAWGUploadException(HDAWGException):
    def __str__(self) -> str:
        return "Upload to the instrument failed."
