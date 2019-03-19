import sys
import functools
from typing import List, Tuple, Set, Callable, Optional, Dict, Mapping, Sequence, NamedTuple
from types import MappingProxyType
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

import gmpy2

import numpy as np
import textwrap
import time

from qupulse.utils.types import ChannelID, TimeType
from qupulse._program._loop import Loop, make_compatible
from qupulse._program.waveforms import Waveform
from qupulse.hardware.awgs.base import AWG, ChannelNotFoundException


# TODO: rename file, make lower case.
# TODO: make format() calls to device node tree more explicit using :d, :.9g for numbers and booleans.

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


class WaveformEntry:
    def __init__(self, name: str, length: int, waveform: Waveform):
        self.name = name
        self.waveform = waveform
        self.length = length


class WaveformDatabase:
    """Stores raw waveform and name relationship. Name is channelpaor ident _ hash liek in tektronix driver. This saves
    sampled values. Waveforms that have a different structure but the same sampled result should reference the same
    sampled data."""
    def __init__(self, entries: Sequence[WaveformEntry] = None):
        if not entries:
            entries = []
        self._waveforms_by_name = {wf.name: wf for wf in entries}
        self._by_name = MappingProxyType(self._waveforms_by_name)

        self._waveforms_by_data = {wf.waveform: wf for wf in entries}
        self._by_data = MappingProxyType(self._waveforms_by_data)

    @property
    def by_name(self) -> Mapping[str, WaveformEntry]:
        return MappingProxyType(self._waveforms_by_name)

    @property
    def by_data(self) -> Mapping[Waveform, WaveformEntry]:
        return MappingProxyType(self._waveforms_by_data)

    def __iter__(self):
        return iter(self._waveforms_by_data.values())

    def __len__(self):
        return len(self._waveforms_by_data)

    def add_waveform(self, name: str, entry: WaveformEntry, overwrite: bool = False):
        if name in self.by_name:
            if not overwrite:
                raise HDAWGRuntimeError('Waveform {} already existing'.format(name))

        self._waveforms_by_data[entry.waveform] = entry
        self._waveforms_by_name[name] = entry

    def pop_waveform(self, name: str) -> WaveformEntry:
        wf = self._waveforms_by_name.pop(name)
        del self._waveforms_by_data[wf.waveform]
        return wf


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

        # Use ordered dict, so index creation for new programs is trivial (also in case of deletions).
        self._known_programs = OrderedDict()  # type: Dict[str, ProgramEntry]
        self._known_waveforms = WaveformDatabase()
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

        Structure of seqc program:
        //////////  qupulse controlled sequence  //////////
        const PROG_SEL = 0; // User register for switching current program.
        const PROG_IDLE = 0; // User register value reserved for idle program.

        // Start of analog waveform definitions.
        wave idle = zeros(16); // Default idle waveform.
        wave w0 = "channel_group_pattern_waveform_hash";
        ...

        // Start of marker waveform definitions.
        wave m0 = "channel_group_pattern_waveform_hash";
        ...

        // Main loop.
        while(true){
            switch(getUserReg(PROG_SEL)){
                case 1: // Program: program_name
                    repeat(n){
                        playWave(w0+m0,w1+m1);
                        ...
                    }
                    waitWave();
                    setUserReg(PROG_SEL, PROG_IDLE);
                ...
                default:
                    playWave(idle, idle);
            }
        }
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

        def generate_program_index() -> int:
            if self._known_programs:
                last_program = next(reversed(self._known_programs))
                last_index = self._known_programs[last_program].index
                return last_index+1
            else:
                return 1  # First index of programs. 0 reserved for idle pulse.

        def program_to_seqc(prog: Loop) -> str:
            """This creates indentation by creating and destroying a lot of strings. Optimization would be to pass this
            as a parameter."""
            if prog.repetition_count > 1:
                template = '  {}'
                yield 'repeat({:d}) {{'.format(prog.repetition_count)
            else:
                template = '{}'

            if program.is_leaf():
                yield template.format(waveform_to_seqc(prog.waveform))
            else:
                for child in prog.children:
                    for line in program_to_seqc(child):
                        yield template.format(line)

            if program.repetition_count > 1:
                yield '}'

        def waveform_to_seqc(waveform: Waveform) -> str:
            """return command that plays the waveform"""
            # TODO: How to work with zero valued waveforms. Reuse zero pulses?

            wf_name = self.identifier + '_' + str(abs(hash(waveform)))
            time_per_sample = 1/self.sample_rate
            sample_times = np.arange(waveform.duration / time_per_sample) * time_per_sample

            for ch in channels:
                voltage = waveform.get_sampled(ch, sample_times)
                self._known_waveforms.add_waveform('{}_{}'.format(wf_name, ch), voltage)

            return 'playWave({}, {});'.format(wf_name, wf_name)

        # Adjust program to fit criteria.
        make_compatible(program,
                        minimal_waveform_length=16,
                        waveform_quantum=16,
                        sample_rate=self.sample_rate)

        self._known_programs[name] = ProgramEntry(program, HDAWGRegisterFunc.PROG_IDLE.value, '\n')

        # TODO: manage _known_programs_register with name to unique integer mapping used to switch programs.
        awg_program = textwrap.dedent("""\
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
                case 1: // Program: program_name
                    repeat(5){
                        playWave(w0+m0,w0+m0);
                    }
                    while(true);
                default:
                    playWave(idle, idle);
            }
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
        self._current_program = None

    def arm(self, name: str) -> None:
        """Load the program 'name' and arm the device for running it. If name is None the awg will "dearm" its current
        program."""
        if not name:
            self.user_register(HDAWGRegisterFunc.PROG_SEL.value, HDAWGRegisterFunc.PROG_IDLE.value)
            self._current_program = None
        else:
            if name not in self.programs:
                raise HDAWGValueError('{} is unknown on {}'.format(name, self.identifier))
            self._current_program = name
            # TODO: Check if this works.
            self.user_register(HDAWGRegisterFunc.PROG_SEL.value, self._known_programs[name].index)

    def run_current_program(self) -> None:
        """Run armed program."""
        # TODO: playWaveDigTrigger() + digital trigger here, alternative implementation.
        if self._current_program:
            if self._current_program not in self.programs:
                raise HDAWGValueError('{} is unknown on {}'.format(self._current_program, self.identifier))
            # TODO: Check if this works.
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
        node_path = '/{}/awgs/{}/enable'.format(self.device.serial, self.awg_group_index)
        if status is not None:
            self.device.api_session.setInt(node_path, int(status))
            self.device.api_session.sync()  # Global sync: Ensure settings have taken effect on the device.
        return bool(self.device.api_session.getInt(node_path))

    def user_register(self, reg: int, value: int = None) -> int:
        """Query user registers (1-16) and optionally set it."""
        if reg not in range(1, 17):
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
