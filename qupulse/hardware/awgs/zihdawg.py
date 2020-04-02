from pathlib import Path
import functools
from typing import Tuple, Set, Callable, Optional, Mapping, NamedTuple, Iterator
from collections import OrderedDict
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

import time

from qupulse.utils.types import ChannelID, TimeType, time_from_float
from qupulse._program._loop import Loop, make_compatible
from qupulse._program.seqc import HDAWGProgramManager, UserRegister
from qupulse.hardware.awgs.base import AWG, ChannelNotFoundException, AWGAmplitudeOffsetHandling
from qupulse.pulses.parameters import ConstantParameter


def valid_channel(function_object):
    """Check if channel is a valid AWG channels. Expects channel to be 2nd argument after self."""
    @functools.wraps(function_object)
    def valid_fn(*args, **kwargs):
        if len(args) < 2:
            raise HDAWGTypeError('Channel is an required argument.')
        channel = args[1]  # Expect channel to be second positional argument after self.
        if channel not in range(1, 9):
            raise ChannelNotFoundException(channel)
        value = function_object(*args, **kwargs)
        return value
    return valid_fn


class HDAWGRepresentation:
    """HDAWGRepresentation represents an HDAWG8 instruments and manages a LabOne data server api session. A data server
    must be running and the device be discoverable. Channels are per default grouped into pairs."""

    def __init__(self, device_serial: str = None,
                 device_interface: str = '1GbE',
                 data_server_addr: str = 'localhost',
                 data_server_port: int = 8004,
                 api_level_number: int = 6,
                 reset: bool = False,
                 timeout: float = 20) -> None:
        """
        :param device_serial:     Device serial that uniquely identifies this device to the LabOne data server
        :param device_interface:  Either '1GbE' for ethernet or 'USB'
        :param data_server_addr:  Data server address. Must be already running. Default: localhost
        :param data_server_port:  Data server port. Default: 8004 for HDAWG, MF and UHF devices
        :param api_level_number:  Version of API to use for the session, higher number, newer. Default: 6 most recent
        :param reset:             Reset device before initialization
        :param timeout:           Timeout in seconds for uploading
        """
        self._api_session = zhinst.ziPython.ziDAQServer(data_server_addr, data_server_port, api_level_number)
        zhinst.utils.api_server_version_check(self.api_session)  # Check equal data server and api version.
        self.api_session.connectDevice(device_serial, device_interface)
        self._dev_ser = device_serial

        if reset:
            # Create a base configuration: Disable all available outputs, awgs, demods, scopes,...
            zhinst.utils.disable_everything(self.api_session, self.serial)

        self._initialize()

        self._channel_pair_AB = HDAWGChannelPair(self, (1, 2), str(self.serial) + '_AB', timeout)
        self._channel_pair_CD = HDAWGChannelPair(self, (3, 4), str(self.serial) + '_CD', timeout)
        self._channel_pair_EF = HDAWGChannelPair(self, (5, 6), str(self.serial) + '_EF', timeout)
        self._channel_pair_GH = HDAWGChannelPair(self, (7, 8), str(self.serial) + '_GH', timeout)

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
        settings.append(['/{}/awgs/*/time'.format(self.serial), 0])  # Maximum sampling rate.
        settings.append(['/{}/sigouts/*/range'.format(self.serial), HDAWGVoltageRange.RNG_1V.value])
        settings.append(['/{}/awgs/*/outputs/*/amplitude'.format(self.serial), 1.0])  # Default amplitude factor 1.0
        settings.append(['/{}/awgs/*/outputs/*/modulation/mode'.format(self.serial), HDAWGModulationMode.OFF.value])
        settings.append(['/{}/awgs/*/userregs/*'.format(self.serial), 0])  # Reset all user registers to 0.
        settings.append(['/{}/awgs/*/single'.format(self.serial), 1])  # Single execution mode of sequence.
        for ch in range(0, 8):  # Route marker 1 signal for each channel to marker output.
            if ch % 2 == 0:
                output = HDAWGTriggerOutSource.OUT_1_MARK_1.value
            else:
                output = HDAWGTriggerOutSource.OUT_2_MARK_1.value
            settings.append(['/{}/triggers/out/{}/source'.format(self.serial, ch), output])

        self.api_session.set(settings)
        self.api_session.sync()  # Global sync: Ensure settings have taken effect on the device.

    def reset(self) -> None:
        zhinst.utils.disable_everything(self.api_session, self.serial)
        self._initialize()
        self.channel_pair_AB.clear()
        self.channel_pair_CD.clear()
        self.channel_pair_EF.clear()
        self.channel_pair_GH.clear()

    @valid_channel
    def offset(self, channel: int, voltage: float = None) -> float:
        """Query channel offset voltage and optionally set it."""
        node_path = '/{}/sigouts/{:d}/offset'.format(self.serial, channel-1)
        if voltage is not None:
            self.api_session.setDouble(node_path, voltage)
            self.api_session.sync()  # Global sync: Ensure settings have taken effect on the device.
        return self.api_session.getDouble(node_path)

    @valid_channel
    def range(self, channel: int, voltage: float = None) -> float:
        """Query channel voltage range and optionally set it. The instruments selects the next higher available range.
        This is the one-sided range Vp. Total range: -Vp...Vp"""
        node_path = '/{}/sigouts/{:d}/range'.format(self.serial, channel-1)
        if voltage is not None:
            self.api_session.setDouble(node_path, voltage)
            self.api_session.sync()  # Global sync: Ensure settings have taken effect on the device.
        return self.api_session.getDouble(node_path)

    @valid_channel
    def output(self, channel: int, status: bool = None) -> bool:
        """Query channel signal output status (enabled/disabled) and optionally set it. Corresponds to front LED."""
        node_path = '/{}/sigouts/{:d}/on'.format(self.serial, channel-1)
        if status is not None:
            self.api_session.setInt(node_path, int(status))
            self.api_session.sync()  # Global sync: Ensure settings have taken effect on the device.
        return bool(self.api_session.getInt(node_path))

    def get_status_table(self):
        """Return node tree of instrument with all important settings, as well as each channel group as tuple."""
        return (self.api_session.get('/{}/*'.format(self.serial)),
                self.channel_pair_AB.awg_module.get('awgModule/*'),
                self.channel_pair_CD.awg_module.get('awgModule/*'),
                self.channel_pair_EF.awg_module.get('awgModule/*'),
                self.channel_pair_GH.awg_module.get('awgModule/*'))


class HDAWGTriggerOutSource(Enum):
    """Assign a signal to a marker output. This is per AWG Core."""
    AWG_TRIG_1 = 0  # Trigger output assigned to AWG trigger 1, controlled by AWG sequencer commands.
    AWG_TRIG_2 = 1  # Trigger output assigned to AWG trigger 2, controlled by AWG sequencer commands.
    AWG_TRIG_3 = 2  # Trigger output assigned to AWG trigger 3, controlled by AWG sequencer commands.
    AWG_TRIG_4 = 3  # Trigger output assigned to AWG trigger 4, controlled by AWG sequencer commands.
    OUT_1_MARK_1 = 4  # Trigger output assigned to output 1 marker 1.
    OUT_1_MARK_2 = 5  # Trigger output assigned to output 1 marker 2.
    OUT_2_MARK_1 = 6  # Trigger output assigned to output 2 marker 1.
    OUT_2_MARK_2 = 7  # Trigger output assigned to output 2 marker 2.
    TRIG_IN_1 = 8  # Trigger output assigned to trigger inout 1.
    TRIG_IN_2 = 9  # Trigger output assigned to trigger inout 2.
    TRIG_IN_3 = 10  # Trigger output assigned to trigger inout 3.
    TRIG_IN_4 = 11  # Trigger output assigned to trigger inout 4.
    TRIG_IN_5 = 12  # Trigger output assigned to trigger inout 5.
    TRIG_IN_6 = 13  # Trigger output assigned to trigger inout 6.
    TRIG_IN_7 = 14  # Trigger output assigned to trigger inout 7.
    TRIG_IN_8 = 15  # Trigger output assigned to trigger inout 8.
    HIGH = 17 # Trigger output is set to high.
    LOW = 18 # Trigger output is set to low.


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


class HDAWGChannelPair(AWG):
    """Represents a channel pair of the Zurich Instruments HDAWG as an independent AWG entity.
    It represents a set of channels that have to have(hardware enforced) the same:
        -control flow
        -sample rate

    It keeps track of the AWG state and manages waveforms and programs on the hardware.
    """

    MIN_WAVEFORM_LEN = 192

    def __init__(self, hdawg_device: HDAWGRepresentation,
                 channels: Tuple[int, int],
                 identifier: str,
                 timeout: float) -> None:
        super().__init__(identifier)
        self._device = weakref.proxy(hdawg_device)

        if channels not in ((1, 2), (3, 4), (5, 6), (7, 8)):
            raise HDAWGValueError('Invalid channel pair: {}'.format(channels))
        self._channels = channels
        self.timeout = timeout

        self._awg_module = self.device.api_session.awgModule()
        self.awg_module.set('awgModule/device', self.device.serial)
        self.awg_module.set('awgModule/index', self.awg_group_index)
        self.awg_module.execute()
        # Seems creating AWG module sets SINGLE (single execution mode of sequence) to 0 per default.
        self.device.api_session.setInt('/{}/awgs/{:d}/single'.format(self.device.serial, self.awg_group_index), 1)

        self._program_manager = HDAWGProgramManager()
        self._required_seqc_source = self._program_manager.to_seqc_program()
        self._uploaded_seqc_source = None
        self._current_program = None  # Currently armed program.

    @property
    def num_channels(self) -> int:
        """Number of channels"""
        return 2

    @property
    def num_markers(self) -> int:
        """Number of marker channels"""
        return 4

    def upload(self, name: str,
               program: Loop,
               channels: Tuple[Optional[ChannelID], Optional[ChannelID]],
               markers: Tuple[Optional[ChannelID], Optional[ChannelID], Optional[ChannelID], Optional[ChannelID]],
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

        Known programs are handled in host memory most of the time. Only when uploading the
        device memory is touched at all.

        Returning from setting user register in seqc can take from 50ms to 60 ms. Fluctuates heavily. Not a good way to
        have deterministic behaviour "setUserReg(PROG_SEL, PROG_IDLE);".
        """
        if len(channels) != self.num_channels:
            raise HDAWGValueError('Channel ID not specified')
        if len(markers) != self.num_markers:
            raise HDAWGValueError('Markers not specified')
        if len(voltage_transformation) != self.num_channels:
            raise HDAWGValueError('Wrong number of voltage transformations')

        if name in self.programs and not force:
            raise HDAWGValueError('{} is already known on {}'.format(name, self.identifier))

        # Go to qupulse nanoseconds time base.
        q_sample_rate = self.sample_rate / 10**9

        # Adjust program to fit criteria.
        make_compatible(program,
                        minimal_waveform_length=self.MIN_WAVEFORM_LEN,
                        waveform_quantum=16,  # 8 samples for single, 4 for dual channel waveforms.
                        sample_rate=q_sample_rate)

        if self._amplitude_offset_handling == AWGAmplitudeOffsetHandling.IGNORE_OFFSET:
            voltage_offsets = (0., 0.)
        elif self._amplitude_offset_handling == AWGAmplitudeOffsetHandling.CONSIDER_OFFSET:
            voltage_offsets = (self._device.offset(self._channels[0]),
                               self._device.offset(self._channels[1]))
        else:
            raise ValueError('{} is invalid as AWGAmplitudeOffsetHandling'.format(self._amplitude_offset_handling))

        amplitudes = self._device.range(self._channels[0]), self._device.range(self._channels[1])

        if name in self._program_manager.programs:
            self._program_manager.remove(name)

        self._program_manager.add_program(name,
                                          program,
                                          channels=channels,
                                          markers=markers,
                                          voltage_transformations=voltage_transformation,
                                          sample_rate=q_sample_rate,
                                          amplitudes=amplitudes,
                                          offsets=voltage_offsets)

        self._required_seqc_source = self._program_manager.to_seqc_program()
        self._program_manager.waveform_memory.sync_to_file_system(Path(self.user_directory).joinpath('awg', 'waves'))

    def _upload_sourcestring(self, sourcestring: str) -> None:
        """Transfer AWG sequencer program as string to HDAWG and block till compilation and upload finish.
        Allows upload without access to data server file system."""
        if not sourcestring:
            raise HDAWGTypeError('sourcestring must not be empty or compilation will not start.')
        logger = logging.getLogger('ziHDAWG')

        # Transfer the AWG sequence program. Compilation starts automatically if sourcestring is set.
        self.awg_module.set('awgModule/compiler/sourcestring', sourcestring)
        self._poll_compile_and_upload_finished(logger)
        self._uploaded_seqc_source = sourcestring

    def _poll_compile_and_upload_finished(self, logger: logging.Logger) -> None:
        """Blocks till compilation on data server and upload to HDAWG succeed,
        if process takes less time than timeout."""
        time_start = time.time()
        logger.info('Compilation started')
        while self.awg_module.getInt('awgModule/compiler/status') == -1:
            time.sleep(0.1)
        if time.time() - time_start > self._timeout:
            raise HDAWGTimeoutError("Compilation timeout out")

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
            if time.time() - time_start > self._timeout:
                raise HDAWGTimeoutError("Upload timeout out")
        logger.info("{} awgModule/progress: {:.2f}".format(i, self.awg_module.getDouble('awgModule/progress')))

        if self.awg_module.getInt('awgModule/elf/status') == 0:
            logger.info('Upload to the instrument successful')
            logger.info('Process took {:.3f} seconds'.format(time.time()-time_start))
        if self.awg_module.getInt('awgModule/elf/status') == 1:
            raise HDAWGUploadException()

    def set_volatile_parameters(self, program_name: str, parameters: Mapping[str, ConstantParameter]):
        """Set the values of parameters which were marked as volatile on program creation."""
        new_register_values = self._program_manager.get_register_values_to_update_volatile_parameters(program_name,
                                                                                                      parameters)
        if self._current_program == program_name:
            for register, value in new_register_values.items():
                self.user_register(register, value)

    def remove(self, name: str) -> None:
        """Remove a program from the AWG.

        Also discards all waveforms referenced only by the program identified by name.

        Args:
            name: The name of the program to remove.
        """
        self._program_manager.remove(name)
        self._required_seqc_source = self._program_manager.to_seqc_program()

    def clear(self) -> None:
        """Removes all programs and waveforms from the AWG.

        Caution: This affects all programs and waveforms on the AWG, not only those uploaded using qupulse!
        """
        self._program_manager.clear()
        self._current_program = None
        self._required_seqc_source = self._program_manager.to_seqc_program()
        self.arm(None)

    def arm(self, name: Optional[str]) -> None:
        """Load the program 'name' and arm the device for running it. If name is None the awg will "dearm" its current
        program.

        Currently hardware triggering is not implemented. The HDAWGProgramManager needs to emit code that calls
        `waitDigTrigger` to do that.
        """
        if self._required_seqc_source != self._uploaded_seqc_source:
            self._upload_sourcestring(self._required_seqc_source)

        self.user_register(self._program_manager.GLOBAL_CONSTS['TRIGGER_REGISTER'], 0)

        if not name:
            self.user_register(self._program_manager.GLOBAL_CONSTS['PROG_SEL_REGISTER'],
                               self._program_manager.GLOBAL_CONSTS['PROG_SEL_NONE'])
            self._current_program = None
        else:
            if name not in self.programs:
                raise HDAWGValueError('{} is unknown on {}'.format(name, self.identifier))
            self._current_program = name

            # set the registers of initial repetition counts
            for register, value in self._program_manager.get_register_values(name).items():
                assert register not in (self._program_manager.GLOBAL_CONSTS['PROG_SEL_REGISTER'],
                                        self._program_manager.GLOBAL_CONSTS['TRIGGER_REGISTER'])
                self.user_register(register, value)

            self.user_register(self._program_manager.GLOBAL_CONSTS['PROG_SEL_REGISTER'],
                               self._program_manager.name_to_index(name) | int(self._program_manager.GLOBAL_CONSTS['NO_RESET_MASK'], 2))
        self.enable(True)

    def run_current_program(self) -> None:
        """Run armed program."""
        if self._current_program is not None:
            if self._current_program not in self.programs:
                raise HDAWGValueError('{} is unknown on {}'.format(self._current_program, self.identifier))
            if not self.enable():
                self.enable(True)
            self.user_register(self._program_manager.GLOBAL_CONSTS['TRIGGER_REGISTER'] + 1, int(self._program_manager.GLOBAL_CONSTS['TRIGGER_RESET_MASK'], 2))
        else:
            raise HDAWGRuntimeError('No program active')

    @property
    def programs(self) -> Set[str]:
        """The set of program names that can currently be executed on the hardware AWG."""
        return set(self._program_manager.programs.keys())

    @property
    def sample_rate(self) -> TimeType:
        """The default sample rate of the AWG channel group."""
        node_path = '/{}/awgs/{}/time'.format(self.device.serial, self.awg_group_index)
        sample_rate_num = self.device.api_session.getInt(node_path)
        node_path = '/{}/system/clocks/sampleclock/freq'.format(self.device.serial)
        sample_clock = self.device.api_session.getDouble(node_path)

        """Calculate exact rational number based on (sample_clock Sa/s) / 2^sample_rate_num. Otherwise numerical
        imprecision will give rise to errors for very long pulses. fractions.Fraction does not accept floating point
        numerator, which sample_clock could potentially be."""
        return time_from_float(sample_clock) / 2 ** sample_rate_num

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
        node_path = '/{}/awgs/{:d}/enable'.format(self.device.serial, self.awg_group_index)
        if status is not None:
            self.device.api_session.setInt(node_path, int(status))
            self.device.api_session.sync()  # Global sync: Ensure settings have taken effect on the device.
        return bool(self.device.api_session.getInt(node_path))

    def user_register(self, reg: UserRegister, value: int = None) -> int:
        """Query user registers (1-16) and optionally set it.

        Args:
            reg: User register. If it is an int, a warning is raised and it is interpreted as a one based index
            value: Value to set

        Returns:
            User Register value after setting it
        """
        if isinstance(reg, int):
            warnings.warn("User register is not a UserRegister instance. It is interpreted as one based index.")
            reg = UserRegister(one_based_value=reg)

        if reg.to_web_interface() not in range(1, 17):
            raise HDAWGValueError('{reg:repr} not a valid (1-16) register.'.format(reg=reg))

        node_path = '/{}/awgs/{:d}/userregs/{:labone}'.format(self.device.serial, self.awg_group_index, reg)
        if value is not None:
            self.device.api_session.setInt(node_path, value)
            self.device.api_session.sync()  # Global sync: Ensure settings have taken effect on the device.
        return self.device.api_session.getInt(node_path)

    def amplitude(self, channel: int, value: float = None) -> float:
        """Query AWG channel amplitude value and optionally set it. Amplitude in units of full scale of the given
         AWG Output. The full scale corresponds to the Range voltage setting of the Signal Outputs."""
        if channel not in (1, 2):
            raise HDAWGValueError('{} not a valid (1-2) channel.'.format(channel))
        node_path = '/{}/awgs/{:d}/outputs/{:d}/amplitude'.format(self.device.serial, self.awg_group_index, channel-1)
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


class HDAWGIOError(HDAWGException, IOError):
    pass


class HDAWGTimeoutError(HDAWGException, TimeoutError):
    pass


class HDAWGCompilationException(HDAWGException):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self) -> str:
        return "Compilation failed: {}".format(self.msg)


class HDAWGUploadException(HDAWGException):
    def __str__(self) -> str:
        return "Upload to the instrument failed."


if __name__ == "__main__":
    from qupulse.pulses import TablePT, SequencePT, RepetitionPT
    hdawg = HDAWGRepresentation(device_serial='dev8075', device_interface='USB')

    entry_list1 = [(0, 0), (20, .2, 'hold'), (40, .3, 'linear'), (80, 0, 'jump')]
    entry_list2 = [(0, 0), (20, -.2, 'hold'), (40, -.3, 'linear'), (50, 0, 'jump')]
    entry_list3 = [(0, 0), (20, -.2, 'linear'), (50, -.3, 'linear'), (70, 0, 'jump')]
    tpt1 = TablePT({0: entry_list1, 1: entry_list2}, measurements=[('m', 20, 30)])
    tpt2 = TablePT({0: entry_list2, 1: entry_list1})
    tpt3 = TablePT({0: entry_list3, 1: entry_list2}, measurements=[('m', 10, 50)])
    rpt = RepetitionPT(tpt1, 4)
    spt = SequencePT(tpt2, rpt)
    rpt2 = RepetitionPT(spt, 2)
    spt2 = SequencePT(rpt2, tpt3)
    p = spt2.create_program()

    ch = (0, 1)
    mk = (0, None)
    vt = (lambda x: x, lambda x: x)
    hdawg.channel_pair_AB.upload('table_pulse_test1', p, ch, mk, vt)

    entry_list_zero = [(0, 0), (100, 0, 'hold')]
    entry_list_step = [(0, 0), (50, .5, 'hold'), (100, 0, 'hold')]
    marker_start = TablePT({'P1': entry_list_zero, 'marker': entry_list_step})
    tpt1 = TablePT({'P1': entry_list_zero, 'marker': entry_list_zero})
    spt2 = SequencePT(marker_start, tpt1)

    p = spt2.create_program()

    ch = ('P1', None)
    mk = ('marker', None)
    voltage_transform = (lambda x: x,) * len(ch)
    hdawg.channel_pair_AB.upload('table_pulse_test2', p, ch, mk, voltage_transform)

