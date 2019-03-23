from pathlib import Path
import functools
from typing import List, Tuple, Set, Callable, Optional, Dict, Mapping, Sequence, NamedTuple, Generator, Iterator
from collections import OrderedDict
from enum import Enum
import weakref
import logging
import warnings
import time

try:
    import zhinst.ziPython
    import zhinst.utils
except ImportError:
    warnings.warn('Zurich Instruments LabOne python API is distributed via the Python Package Index. Install with pip.')
    raise

import gmpy2

import numpy as np
import textwrap
import time

from qupulse.utils.types import ChannelID, TimeType
from qupulse._program._loop import Loop, make_compatible
from qupulse._program.waveforms import Waveform
from qupulse.hardware.awgs.base import AWG, ChannelNotFoundException


# TODO: rename file, make lower case.

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


class HDAWGRegisterFunc(Enum):
    """Functions of registers for sequence control."""
    PROG_SEL = 1  # Use this register to select which program in the sequence should be running.
    PROG_IDLE = 0  # This value of the PROG_SEL register is reserved for the idle waveform.


class ProgramEntry(NamedTuple):
    """Entry of known AWG programs."""
    program: Loop
    index: int  # Program to seqc switch case mapping.
    seqc_rep: str  # Seqc representation of program inside case statement.


class HDAWGChannelPair(AWG):
    """Represents a channel pair of the Zurich Instruments HDAWG as an independent AWG entity.
    It represents a set of channels that have to have(hardware enforced) the same:
        -control flow
        -sample rate

    It keeps track of the AWG state and manages waveforms and programs on the hardware.
    """

    def __init__(self, hdawg_device: HDAWGRepresentation, channels: Tuple[int, int], identifier: str, timeout: float) -> None:
        super().__init__(identifier)
        self._device = weakref.proxy(hdawg_device)

        if channels not in ((1, 2), (3, 4), (5, 6), (7, 8)):
            raise HDAWGValueError('Invalid channel pair: {}'.format(channels))
        self._channels = channels
        self._timeout = timeout

        self._awg_module = self.device.api_session.awgModule()
        self.awg_module.set('awgModule/device', self.device.serial)
        self.awg_module.set('awgModule/index', self.awg_group_index)
        self.awg_module.execute()
        # Seems creating AWG module sets SINGLE (single execution mode of sequence) to 0 per default.
        self.device.api_session.setInt('/{}/awgs/{:d}/single'.format(self.device.serial, self.awg_group_index), 1)

        # Use ordered dict, so index creation for new programs is trivial (also in case of deletions).
        self._known_programs = OrderedDict()  # type: Dict[str, ProgramEntry]
        self._wave_manager = HDAWGWaveManager(self.user_directory, self.identifier)
        self._current_program = None  # Currently armed program.

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

        self._known_programs is dict of known programs. Handled in host memory most of the time. Only when uploading the
        device memory is touched at all.

        Upload steps:
        1. All necessary waveforms are sampled with the current device sample rate.
        2. All files matching pattern of this channel group are deleted from waves folder.
        3. The channel & marker waveforms are saved as csv-files in the wave folder with this channel group prefix.
        4. _known_programs is indexed and index is stored with each program. Used later to identify program by user reg.
        5. _known_programs dict is translated to seqc program and saved as seqc-file in src folder. Overwrites old file.
        6. AWG sequencer is disabled.
        7. seqc program file compilation and upload is started.
        8. Program select user register is set to idle program.
        8. AWG sequencer is enabled.

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

        def generate_program_index() -> int:
            if self._known_programs:
                last_program = next(reversed(self._known_programs))
                last_index = self._known_programs[last_program].index
                return last_index+1
            else:
                return 1  # First index of programs. 0 reserved for idle pulse.

        def program_to_seqc(prog: Loop) -> Iterator[str]:
            """This creates indentation by creating and destroying a lot of strings. Optimization would be to pass this
            as a parameter."""
            if prog.repetition_count > 1:
                template = '  {}'
                yield 'repeat({:d}) {{'.format(prog.repetition_count)
            else:
                template = '{}'

            if prog.is_leaf():
                yield template.format(waveform_to_seqc(prog.waveform))
            else:
                for child in prog.children:
                    for line in program_to_seqc(child):
                        yield template.format(line)

            if prog.repetition_count > 1:
                yield '}'

        def waveform_to_seqc(waveform: Waveform) -> str:
            """return command that plays the waveform"""
            wf_name, mk_name = self._wave_manager.register(waveform,
                                                           channels,
                                                           markers,
                                                           voltage_transformation,
                                                           self.sample_rate,
                                                           force)
            if mk_name is None:
                return 'playWave("{}");'.format(wf_name)
            else:
                return 'playWave(add("{}", "{}"));'.format(wf_name, mk_name)

        def case_wrap_program(prog: ProgramEntry, prog_name, indent = 8) -> str:
            indented_seqc = textwrap.indent('{}\nwhile(true);'.format(prog.seqc_rep), ' ' * 4)
            case_str = 'case {:d}: // Program name: {}\n{}'.format(prog.index, prog_name, indented_seqc)
            return textwrap.indent(case_str, ' ' * indent)

        # Adjust program to fit criteria.
        make_compatible(program,
                        minimal_waveform_length=16,
                        waveform_quantum=8,  # 8 samples for single, 4 for dual channel waaveforms.
                        sample_rate=self.sample_rate)

        seqc_gen = program_to_seqc(program)
        self._known_programs[name] = ProgramEntry(program, generate_program_index(), '\n'.join(seqc_gen))

        awg_sequence = textwrap.dedent("""\
        //////////  qupulse sequence (_upload_time_) //////////
        
        const PROG_SEL = 0; // User register for switching current program.

        // Start of analog waveform definitions.
        wave idle = zeros(16); // Default idle waveform.
        _analog_waveform_block_

        // Start of marker waveform definitions.
        _marker_waveform_block_

        // Arm program switch.
        var prog_sel = getUserReg(PROG_SEL);
        
        // Main loop.
        while(true){
            switch(prog_sel){
        _case_block_
                default:
                    playWave(idle, idle);
            }
        }
        """)
        awg_sequence = awg_sequence.replace('_upload_time_', time.strftime('%c'))
        awg_sequence = awg_sequence.replace('_analog_waveform_block_', '')
        awg_sequence = awg_sequence.replace('_marker_waveform_block_', '')

        case_block = []
        for name, entry in self._known_programs.items():
            case_block.append(case_wrap_program(entry, name))

        awg_sequence = awg_sequence.replace('_case_block_', '\n'.join(case_block))

        self._upload_sourcestring(awg_sequence)

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

    def remove(self, name: str) -> None:
        """Remove a program from the AWG.

        Also discards all waveforms referenced only by the program identified by name.

        Args:
            name: The name of the program to remove.
        """
        # TODO: Call removal of program waveforms on WaveManger.
        self._known_programs.pop(name)

    def clear(self) -> None:
        """Removes all programs and waveforms from the AWG.

        Caution: This affects all programs and waveforms on the AWG, not only those uploaded using qupulse!
        """
        self._known_programs.clear()
        self._wave_manager.clear()
        self._current_program = None

    def arm(self, name: Optional[str]) -> None:
        """Load the program 'name' and arm the device for running it. If name is None the awg will "dearm" its current
        program."""
        if not name:
            self.user_register(HDAWGRegisterFunc.PROG_SEL.value, HDAWGRegisterFunc.PROG_IDLE.value)
            self._current_program = None
        else:
            if name not in self.programs:
                raise HDAWGValueError('{} is unknown on {}'.format(name, self.identifier))
            self._current_program = name
            self.user_register(HDAWGRegisterFunc.PROG_SEL.value, self._known_programs[name].index)

    def run_current_program(self) -> None:
        """Run armed program."""
        # TODO: playWaveDigTrigger() + digital trigger here, alternative implementation.
        if self._current_program is not None:
            if self._current_program not in self.programs:
                raise HDAWGValueError('{} is unknown on {}'.format(self._current_program, self.identifier))
            if self.enable():
                self.enable(False)
            self.enable(True)
        else:
            raise HDAWGRuntimeError('No program active')

    @property
    def programs(self) -> Set[str]:
        """The set of program names that can currently be executed on the hardware AWG."""
        return set(self._known_programs.keys())

    @property
    def sample_rate(self) -> gmpy2.mpq:
        """The default sample rate of the AWG channel group."""
        node_path = '/{}/awgs/{}/time'.format(self.device.serial, self.awg_group_index)
        sample_rate_num = self.device.api_session.getInt(node_path)
        node_path = '/{}/system/clocks/sampleclock/freq'.format(self.device.serial)
        sample_clock = self.device.api_session.getDouble(node_path)

        """Calculate exact rational number based on (sample_clock Sa/s) / 2^sample_rate_num. Otherwise numerical
        imprecision will give rise to errors for very long pulses. fractions.Fraction does not accept floating point
        numerator, which sample_clock could potentially be."""
        return gmpy2.mpq(sample_clock, 2 ** sample_rate_num)

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

    def user_register(self, reg: int, value: int = None) -> int:
        """Query user registers (1-16) and optionally set it."""
        if reg not in range(1, 17):
            raise HDAWGValueError('{} not a valid (1-16) register.'.format(reg))
        node_path = '/{}/awgs/{:d}/userregs/{:d}'.format(self.device.serial, self.awg_group_index, reg-1)
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


class HDAWGWaveManager:
    """Manages waveforms and IO of sampled data to disk."""
    # TODO: Manage references and delete csv file when no program uses it.
    # TODO: Implement voltage transformation.
    # TODO: Voltage to -1..1 range and check if max amplitude in range+offset window.
    # TODO: Manage side effects if reusing data over several programs and a shared waveform is overwritten.

    class WaveformEntry(NamedTuple):
        """Entry of known waveforms."""
        waveform: Waveform
        marker_name: Optional[str]  # None if this waveform does not define any markers.

    def __init__(self, user_dir: str, awg_identifier: str) -> None:
        self._known_waveforms = dict()  # type: Dict[str, HDAWGWaveManager.WaveformEntry]
        self._by_data = dict()  # type: Dict[int, str]
        self._file_type = 'csv'
        self._awg_prefix = awg_identifier
        self._wave_dir = Path(user_dir).joinpath('awg', 'waves')
        if not self._wave_dir.is_dir():
            raise HDAWGIOError('{} does not exist or is not a directory'.format(self._wave_dir))
        self.clear()

    def clear(self) -> None:
        self._known_waveforms.clear()
        self._by_data.clear()
        for wave_file in self._wave_dir.glob(self._awg_prefix + '_*.' + self._file_type):
            wave_file.unlink()

    def remove(self, name: str) -> None:
        self._known_waveforms.pop(name)
        for wf_entry, wf_name in self._by_data.items():
            if wf_name == name:
                del self._by_data[wf_entry]
                break
        wave_path = self.full_file_path(name)
        wave_path.unlink()

    def full_file_path(self, name: str) -> Path:
        return self._wave_dir.joinpath(name + '.' + self._file_type)

    def generate_name(self, waveform: Waveform) -> str:
        return self._awg_prefix + '_' + str(abs(hash(waveform)))

    def to_file(self, name: str, wave_data: np.ndarray, fmt: str = '%f', overwrite: bool = False) -> None:
        file_path = self.full_file_path(name)
        if file_path.is_file() and not overwrite:
            raise HDAWGIOError('{} already exists'.format(file_path))
        np.savetxt(file_path, wave_data, fmt=fmt, delimiter=' ')

    def calc_hash(self, data: np.ndarray) -> int:
        return hash(bytes(data))

    def register(self, waveform: Waveform,
                 channels: Tuple[Optional[ChannelID], Optional[ChannelID]],
                 markers: Tuple[Optional[ChannelID], Optional[ChannelID]],
                 voltage_transformation: Tuple[Callable, Callable],
                 sample_rate: gmpy2.mpq,
                 overwrite: bool = False) -> Tuple[str, Optional[str]]:
        name = self.generate_name(waveform)
        if name in self._known_waveforms:
            return name, self._known_waveforms[name].marker_name

        time_per_sample = 1/sample_rate
        sample_times = np.arange(waveform.duration / time_per_sample) * time_per_sample

        voltage = np.zeros((len(sample_times), 2), dtype=float)
        for idx, chan in enumerate(channels):
            if chan is not None:
                voltage[:, idx] = waveform.get_sampled(chan, sample_times)

        # Reuse sampled data, if available.
        voltage_hash = self.calc_hash(voltage)
        if voltage_hash in self._by_data:
            name = self._by_data[voltage_hash]
        else:
            self._by_data[voltage_hash] = name
            self.to_file(name, voltage, overwrite=overwrite)

        if markers[0] is not None or markers[1] is not None:
            marker_name = name + '_m'

            marker_output = np.zeros((len(sample_times), 2), dtype=np.uint8)
            for idx, marker in enumerate(markers):
                if marker is not None:
                    # TODO: Implement correct marker generation.
                    temp = np.tile(np.vstack((np.ones((64, 1)), np.zeros((64, 1)))),
                                   (len(sample_times)//64, 1))
                    marker_output[:, idx] = temp[:len(sample_times)].ravel()

            # Reuse sampled data, if available.
            marker_hash = self.calc_hash(marker_output)
            if marker_hash in self._by_data:
                marker_name = self._by_data[marker_hash]
            else:
                self._by_data[marker_hash] = marker_name
                self.to_file(marker_name, marker_output, fmt='%d', overwrite=overwrite)
        else:
            marker_name = None

        self._known_waveforms[name] = self.WaveformEntry(waveform, marker_name)
        return name, marker_name


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
    entry_list1 = [(0, 0), (20e-9, .2, 'hold'), (40e-9, .3, 'linear'), (60e-9, 0, 'jump')]
    entry_list2 = [(0, 0), (20e-9, -.2, 'hold'), (40e-9, -.3, 'linear'), (60e-9, 0, 'jump')]
    entry_list3 = [(0, 0), (20e-9, -.2, 'linear'), (50e-9, -.3, 'linear'), (60e-9, 0, 'jump')]
    tpt1 = TablePT({0: entry_list1, 1: entry_list2}, measurements=[('m', 20e-9, 30e-9)])
    tpt2 = TablePT({0: entry_list2, 1: entry_list1}, measurements=[('m', 20e-9, 30e-9)])
    tpt3 = TablePT({0: entry_list3, 1: entry_list2}, measurements=[('m', 20e-9, 30e-9)])
    rpt = RepetitionPT(tpt1, 4)
    spt = SequencePT(tpt2, rpt)
    rpt2 = RepetitionPT(spt, 2)
    spt2 = SequencePT(rpt2, tpt3)
    p = spt2.create_program()

    ch = (0, 1)
    mk = (0, None)
    vt = (None, None)
    hdawg = HDAWGRepresentation(device_serial='dev8075', device_interface='USB')
    hdawg.channel_pair_AB.upload('table_pulse_test', p, ch, mk, vt)

