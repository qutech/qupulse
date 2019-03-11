import sys
import functools
from typing import List, Tuple, Set, Callable, Optional, Dict
from enum import Enum
import weakref
import logging
import warnings

try:
    import zhinst.ziPython
    import zhinst.utils
except ImportError:
    warnings.warn('Zurich Instruments LabOne python API is distributed via the Python Package Index. Install with pip.')
    raise

import numpy as np
import textwrap
import time

from qupulse.utils.types import ChannelID, TimeType
from qupulse._program._loop import Loop, make_compatible
from qupulse._program.waveforms import Waveform
from qupulse.hardware.awgs.base import AWG, ChannelNotFoundException


def valid_channel(function_object):
    """Check if channel is a valid AWG channels. Expects channel to be 2nd argument after self."""
    @functools.wraps(function_object)
    def valid_fn(*args, **kwargs):
        if len(args) < 2:
            raise HDAWGTypeError('Channel is an required argument.')
        channel = args[1]  # Expect channel to be second positional argument after self.
        if channel not in range(1, 8):
            raise ChannelNotFoundException(channel)
        value = function_object(*args, **kwargs)
        return value
    return valid_fn


class WaveformDatabase:
    def get_name(self, waveform: Waveform, sample_rate):
        raise NotImplementedError()


def waveform_to_seqc(waveform: Waveform, channels: Tuple[Optional[ChannelID], Optional[ChannelID]], wf_database: WaveformDatabase, sample_rate: TimeType):
    """return command that plays the waveform"""

    sample_times = np.arange(waveform.duration / sample_rate) *

    for ch in channels:
        waveform.get_sampled(ch, )

    wf_name = wf_database.get_name(waveform, sample_rate)

    raise NotImplementedError()


def program_to_seqc(program: Loop, channels: Tuple[Optional[ChannelID], Optional[ChannelID]], sample_rate, wf_database):
    """This creates indentation by creating and destroying a lot of strings. Optimization would be to pass this as a
    parameter"""
    if program.repetition_count > 1:
        template = '  %s'
        yield 'repeat(%d) {' % program.repetition_count
    else:
        template = '%s'

    if program.is_leaf():
        yield template % waveform_to_seqc(program.waveform, channels, wf_database)
    else:
        for child in program.children:
            for line in program_to_seqc(child, channels, sample_rate, wf_database):
                yield template % line

    if program.repetition_count > 1:
        yield '}'




class HDAWGRepresentation:
    """HDAWGRepresentation represents an HDAWG8 instruments and manages a LabOne data server api session. A data server
    must be running and the device be discoverable. Channels are per default grouped into pairs."""

    def __init__(self, device_serial=None,
                 device_interface='1GbE',
                 data_server_addr='localhost',
                 data_server_port=8004,
                 api_level_number=6,
                 reset=False):
        """
        :param device_serial:     Device serial that uniquely identifies this device to the LabOne data server
        :param device_interface:  Either '1GbE' for ethernet or 'USB'
        :param data_server_addr:  Data server address. Must be already running. Default: localhost
        :param data_server_port:  Data server port. Default: 8004 for HDAWG, MF and UHF devices
        :param api_level_number:  Version of API to use for the session, higher number, newer. Default: 6 most recent
        :param reset:             Reset device before initialization
        """
        self._api_session = zhinst.ziPython.ziDAQServer(data_server_addr, data_server_port, api_level_number)
        zhinst.utils.api_server_version_check(self.api_session)  # Check equal data server and api version.
        self.api_session.connectDevice(device_serial, device_interface)
        self._dev_ser = device_serial

        if reset:
            # TODO: Check if utils function is sufficient, or a custom reset function is required.
            # Create a base configuration: Disable all available outputs, awgs, demods, scopes,...
            zhinst.utils.disable_everything(self.api_session, self.serial)

        self._initialize()

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

    def _initialize(self) -> None:
        settings = []
        settings.append(['/{}/system/awg/channelgrouping'.format(self.serial),
                         HDAWGChannelGrouping.CHAN_GROUP_4x2.value])
        settings.append(['/{}/awgs/*/time'.format(self.serial), HDAWGSamplingRate.AWG_RATE_2400MHZ.value])
        settings.append(['/{}/sigouts/*/range'.format(self.serial), HDAWGVoltageRange.RNG_1V.value])
        settings.append(['/{}/awgs/*/outputs/*/amplitude'.format(self.serial), 1.0])  # Default amplitude factor 1.0
        settings.append(['/{}/awgs/*/outputs/*/modulation/mode'.format(self.serial), HDAWGModulationMode.OFF.value])
        settings.append(['/{}/awgs/*/userregs/*'.format(self.serial), 0])  # Reset all user registers to 0.
        settings.append(['/{}/awgs/*/single'.format(self.serial), 1])  # Single execution mode of sequence.

        self.api_session.set(settings)
        self.api_session.sync()  # Global sync: Ensure settings have taken effect on the device.

    def reset(self) -> None:
        # TODO: Check if utils function is sufficient to reset device.
        zhinst.utils.disable_everything(self.api_session, self.serial)
        self._initialize()
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

    def get_status_table(self):
        """Return node tree of instrument with all important settings, as well as each channel group as tuple."""
        return (self.api_session.get('/{}/*'.format(self.serial)),
                self.channel_pair_AB.awg_module.get('awgModule/*'),
                self.channel_pair_CD.awg_module.get('awgModule/*'),
                self.channel_pair_EF.awg_module.get('awgModule/*'),
                self.channel_pair_GH.awg_module.get('awgModule/*'))


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


class HDAWGRegisterFunc(Enum):
    """Functions of registers for sequence control."""
    PROG_SEL = 1  # Use this register to select which program in the sequence should be running.


class HDAWGChannelPair(AWG):
    """Represents a channel pair of the Zurich Instruments HDAWG as an independent AWG entity.
    It represents a set of channels that have to have(hardware enforced) the same:
        -control flow
        -sample rate

    It keeps track of the AWG state and manages waveforms and programs on the hardware.
    """

    def __init__(self, hdawg_device: HDAWGRepresentation, channels: Tuple[int, int], identifier: str):
        super().__init__(identifier)
        self._device = weakref.proxy(hdawg_device)

        if channels not in ((1, 2), (3, 4), (5, 6), (7, 8)):
            raise HDAWGValueError('Invalid channel pair: {}'.format(channels))
        self._channels = channels

        self._awg_module = self.device.api_session.awgModule()
        self.awg_module.set('awgModule/device', self.device.serial)
        self.awg_module.set('awgModule/index', self.awg_group_index)
        self.awg_module.execute()
        # Seems creating AWG module sets SINGLE (single execution mode of sequence) to 0 per default.
        self.device.api_session.setInt('/{}/awgs/{}/single'.format(self.device.serial, self.awg_group_index), 1)

        self._known_programs = dict()  # type: Dict[str, Loop]
        self._known_programs_register = dict()  # type: Dict[str, int]
        self._current_program = None

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
            raise HDAWGValueError('Channel ID not specified')
        if len(markers) != self.num_markers:
            raise HDAWGValueError('Markers not specified')
        if len(voltage_transformation) != self.num_channels:
            raise HDAWGValueError('Wrong number of voltage transformations')

        if name in self.programs:
            if not force:
                raise HDAWGValueError('{} is already known on {}'.format(name, self.identifier))

        # Adjust program to fit criteria.
        make_compatible(program,
                        minimal_waveform_length=16,
                        waveform_quantum=16,
                        sample_rate=self.sample_rate)

        self._known_programs[name] = program

        # TODO: user registers could be used to arm programs. Uploaded sequence would have a switch statement.
        # TODO: Create seqc sourcestring out of all known programs and upload everything.
        # TODO: manage _known_programs_register with name to unique integer mapping used to switch programs.
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

        self._upload_sourcestring(awg_program)

    # TODO: test this function. must waveform be seperated by only comma or comma space?
    # TODO: is waveform list really necessary? In example: "All CSV files within the waves directory are automatically recognized by the AWG module"
    def _upload_sourcefile(self, sourcefile: str, csv_waveforms: List[str] = None) -> None:
        """Transfer AWG sequencer program as file to HDAWG and block till compilation and upload finish. Sourcefile must
        reside in the LabOne user directory + "awg/src" and optional csv file waveforms in  "awg/waves"."""
        if not sourcefile:
            raise HDAWGTypeError('sourcefile must not empty.')
        if csv_waveforms is None:
            csv_waveforms = []
        logger = logging.getLogger('ziHDAWG')

        self.awg_module.set('awgModule/compiler/waveforms', ','.join(csv_waveforms))
        self.awg_module.set('awgModule/compiler/sourcefile', sourcefile)
        self.awg_module.set('awgModule/compiler/start', 1)  # Compilation must be started manually in case of file.
        self._poll_compile_and_upload_finished(logger)

    # TODO: add csv waveform variable? Does this behave different than sourcefile? Only automatic recognized if sourcestring?
    def _upload_sourcestring(self, sourcestring: str) -> None:
        """Transfer AWG sequencer program as string to HDAWG and block till compilation and upload finish.
        Allows upload without access to data server file system."""
        if not sourcestring:
            raise HDAWGTypeError('sourcestring must not be empty or compilation will not start.')
        logger = logging.getLogger('ziHDAWG')

        # Transfer the AWG sequence program. Compilation starts automatically if sourcestring is set.
        self.awg_module.set('awgModule/compiler/sourcestring', sourcestring)
        self._poll_compile_and_upload_finished(logger)

    def _poll_compile_and_upload_finished(self, logger: logging.Logger) -> None:
        """Blocks till compilation on data server and upload to HDAWG succeed."""
        logger.info('Compilation started')
        while self.awg_module.getInt('awgModule/compiler/status') == -1:
            time.sleep(0.1)

        if self.awg_module.getInt('awgModule/compiler/status') == 1:
            msg = self.awg_module.getString('awgModule/compiler/statusstring')
            logger.error(msg)
            raise HDAWGCompilationException(msg)

        if self.awg_module.getInt('awgModule/compiler/status') == 0:
            logger.info('Compilation successful')
        if self.awg_module.getInt('awgModule/compiler/status') == 2:
            msg = self.awg_module.getString('awgModule/compiler/statusstring')
            logger.warning(msg)

        i = 0
        while ((self.awg_module.getDouble('awgModule/progress') < 1.0) and
               (self.awg_module.getInt('awgModule/elf/status') != 1)):
            time.sleep(0.2)
            logger.info("{} awgModule/progress: {:.2f}".format(i, self.awg_module.getDouble('awgModule/progress')))
            i = i + 1
        logger.info("{} awgModule/progress: {:.2f}".format(i, self.awg_module.getDouble('awgModule/progress')))

        if self.awg_module.getInt('awgModule/elf/status') == 0:
            logger.info('Upload to the instrument successful')
        if self.awg_module.getInt('awgModule/elf/status') == 1:
            raise HDAWGUploadException()

    def remove(self, name: str) -> None:
        """Remove a program from the AWG.

        Also discards all waveforms referenced only by the program identified by name.

        Args:
            name: The name of the program to remove.
        """
        # TODO: Can this function be implemented with the HDAWG as intended?
        self._known_programs.pop(name)

    def clear(self) -> None:
        """Removes all programs and waveforms from the AWG.

        Caution: This affects all programs and waveforms on the AWG, not only those uploaded using qupulse!
        """
        # TODO: Can this function be implemented with the HDAWG as intended?
        self._known_programs.clear()
        self._known_programs_register.clear()
        self._current_program = None

    def arm(self, name: str) -> None:
        """Load the program 'name' and arm the device for running it. If name is None the awg will "dearm" its current
        program."""
        if name not in self.programs:
            raise HDAWGValueError('{} is unknown on {}'.format(name, self.identifier))
        self._current_program = name
        # TODO: Check if this works.
        self.user_register(HDAWGRegisterFunc.PROG_SEL.value, self._known_programs_register[name])

    def run_current_program(self) -> None:
        """Run armed program."""
        if self._current_program:
            if self._current_program not in self.programs:
                raise HDAWGValueError('{} is unknown on {}'.format(self._current_program, self.identifier))
            # TODO: Check if this works.
            self.enable(True)
        else:
            raise HDAWGRuntimeError('No program active')

    @property
    def programs(self) -> Set[str]:
        """The set of program names that can currently be executed on the hardware AWG."""
        return set(self._known_programs.keys())

    @property
    def sample_rate(self) -> float:
        """The default sample rate of the AWG channel group."""
        node_path = '/{}/awgs/{}/time'.format(self.device.serial, self.awg_group_index)
        sample_rate_num = self.device.api_session.getInt(node_path)
        return HDAWGSamplingRate(sample_rate_num).exact_rate()

    @property
    def awg_group_index(self) -> int:
        """AWG node group index assuming 4x2 channel grouping. Then 0...3 will give appropriate index of group."""
        return self._channels[0] // 2

    @property
    def device(self) -> HDAWGRepresentation:
        """Reference to HDAWG representation."""
        return self._device

    @property
    def awg_module(self) -> zhinst.ziPython.AwgModule:
        """Each AWG channel group has its own awg module to manage program compilation and upload."""
        return self._awg_module

    @property
    def user_directory(self) -> str:
        """LabOne user directory with subdirectories: "awg/src" (seqc sourcefiles), "awg/elf" (compiled AWG binaries),
        "awag/waves" (user defined csv waveforms)."""
        return self.awg_module.getString('awgModule/directory')

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
            raise HDAWGValueError('{} not a valid (1-16) register.'.format(reg))
        node_path = '/{}/awgs/{}/userregs/{}'.format(self.device.serial, self.awg_group_index, reg-1)
        if value is not None:
            self.device.api_session.setInt(node_path, value)
            self.device.api_session.sync()  # Global sync: Ensure settings have taken effect on the device.
        return self.device.api_session.getInt(node_path)

    def amplitude(self, channel: int, value: float = None) -> float:
        """Query AWG channel amplitude value and optionally set it. Amplitude in units of full scale of the given
         AWG Output. The full scale corresponds to the Range voltage setting of the Signal Outputs."""
        if channel not in (1, 2):
            raise HDAWGValueError('{} not a valid (1-2) channel.'.format(channel))
        node_path = '/{}/awgs/{}/outputs/{}/amplitude'.format(self.device.serial, self.awg_group_index, channel-1)
        if value is not None:
            self.device.api_session.setDouble(node_path, value)
            self.device.api_session.sync()  # Global sync: Ensure settings have taken effect on the device.
        return self.device.api_session.getDouble(node_path)


class HDAWGException(Exception):
    """Base exception class for HDAWG errors."""
    pass


class HDAWGValueError(HDAWGException, ValueError):
    pass


class HDAWGTypeError(HDAWGException, TypeError):
    pass


class HDAWGRuntimeError(HDAWGException, RuntimeError):
    pass


class HDAWGCompilationException(HDAWGException):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self) -> str:
        return "Compilation failed: {}".format(self.msg)


class HDAWGUploadException(HDAWGException):
    def __str__(self) -> str:
        return "Upload to the instrument failed."
