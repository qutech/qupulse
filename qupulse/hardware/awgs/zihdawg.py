from pathlib import Path
import functools
from typing import Tuple, Set, Callable, Optional, Mapping, Generator, Union, Sequence, Dict
from enum import Enum
import weakref
import logging
import warnings
import pathlib
import hashlib
import argparse
import re
from abc import abstractmethod

try:
    # zhinst fires a DeprecationWarning from its own code in some versions...
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        import zhinst.utils
except ImportError:
    warnings.warn('Zurich Instruments LabOne python API is distributed via the Python Package Index. Install with pip.')
    raise

try:
    from zhinst import core as zhinst_core
except ImportError:
    # backward compability
    from zhinst import ziPython as zhinst_core

import time

from qupulse.utils.types import ChannelID, TimeType, time_from_float
from qupulse._program._loop import Loop, make_compatible
from qupulse._program.seqc import HDAWGProgramManager, UserRegister, WaveformFileSystem
from qupulse.hardware.awgs.base import AWG, ChannelNotFoundException, AWGAmplitudeOffsetHandling
from qupulse.pulses.parameters import ConstantParameter
from qupulse.hardware.util import traced


logger = logging.getLogger('qupulse.hdawg')


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


def _amplitude_scales(api_session, serial: str):
    return tuple(
        api_session.getDouble(f'/{serial}/awgs/{ch // 2:d}/outputs/{ch % 2:d}/amplitude')
        for ch in range(8)
    )

def _sigout_double(api_session, prop: str, serial: str, channel: int, value: float = None) -> float:
    """Query channel offset voltage and optionally set it."""
    node_path = f'/{serial}/sigouts/{channel-1:d}/{prop}'
    if value is not None:
        api_session.setDouble(node_path, value)
        api_session.sync()  # Global sync: Ensure settings have taken effect on the device.
    return api_session.getDouble(node_path)

def _sigout_range(api_session, serial: str, channel: int, voltage: float = None) -> float:
    return _sigout_double(api_session, 'range', serial, channel, voltage)

def _sigout_offset(api_session, serial: str, channel: int, voltage: float = None) -> float:
    return _sigout_double(api_session, 'offset', serial, channel, voltage)

def _sigout_on(api_session, serial: str, channel: int, value: bool = None) -> bool:
    """Query channel signal output status (enabled/disabled) and optionally set it. Corresponds to front LED."""
    node_path = f'/{serial}/sigouts/{channel-1:d}/on'
    if value is not None:
        api_session.setInt(node_path, value)
        api_session.sync()  # Global sync: Ensure settings have taken effect on the device.
    return bool(api_session.getInt(node_path))


@traced
class HDAWGRepresentation:
    """HDAWGRepresentation represents an HDAWG8 instruments and manages a LabOne data server api session. A data server
    must be running and the device be discoverable. Channels are per default grouped into pairs."""

    def __init__(self, device_serial: str = None,
                 device_interface: str = '1GbE',
                 data_server_addr: str = 'localhost',
                 data_server_port: int = 8004,
                 api_level_number: int = 6,
                 reset: bool = False,
                 timeout: float = 20,
                 grouping: 'HDAWGChannelGrouping' = None) -> None:
        """
        :param device_serial:     Device serial that uniquely identifies this device to the LabOne data server
        :param device_interface:  Either '1GbE' for ethernet or 'USB'
        :param data_server_addr:  Data server address. Must be already running. Default: localhost
        :param data_server_port:  Data server port. Default: 8004 for HDAWG, MF and UHF devices
        :param api_level_number:  Version of API to use for the session, higher number, newer. Default: 6 most recent
        :param reset:             Reset device before initialization
        :param timeout:           Timeout in seconds for uploading
        """
        self._api_session = zhinst_core.ziDAQServer(data_server_addr, data_server_port, api_level_number)
        assert zhinst.utils.api_server_version_check(self.api_session)  # Check equal data server and api version.
        self.api_session.connectDevice(device_serial, device_interface)
        self.default_timeout = timeout
        self._dev_ser = device_serial

        if reset:
            # Create a base configuration: Disable all available outputs, awgs, demods, scopes,...
            zhinst.utils.disable_everything(self.api_session, self.serial)

        self._initialize()

        waveform_path = pathlib.Path(self.api_session.awgModule().getString('directory'), 'awg', 'waves')
        self._waveform_file_system = WaveformFileSystem.get_waveform_file_system(waveform_path)
        self._channel_groups: Dict[HDAWGChannelGrouping, Tuple[HDAWGChannelGroup, ...]] = {}

        # TODO: lookup method to find channel count
        n_channels = 8

        for grouping in HDAWGChannelGrouping:
            group_size = grouping.group_size()
            if group_size is None:
                # MDS
                groups = [
                    MDSChannelGroup(self.group_name(0, None), self.default_timeout)
                ]
            else:
                groups = []
                for group_idx in range(n_channels // group_size):
                    groups.append(SingleDeviceChannelGroup(group_idx, group_size,
                                                           identifier=self.group_name(group_idx, group_size),
                                                           timeout=self.default_timeout))
            self._channel_groups[grouping] = tuple(groups)

        if grouping is None:
            grouping = self.channel_grouping
        # activates channel groups
        self.channel_grouping = grouping

    @property
    def waveform_file_system(self) -> WaveformFileSystem:
        return self._waveform_file_system

    @property
    def channel_tuples(self) -> Tuple['HDAWGChannelGroup', ...]:
        return self._get_groups(self.channel_grouping)

    @property
    def channel_pair_AB(self) -> 'HDAWGChannelGroup':
        return self._channel_groups[HDAWGChannelGrouping.CHAN_GROUP_4x2][0]

    @property
    def channel_pair_CD(self) -> 'HDAWGChannelGroup':
        return self._channel_groups[HDAWGChannelGrouping.CHAN_GROUP_4x2][1]

    @property
    def channel_pair_EF(self) -> 'HDAWGChannelGroup':
        return self._channel_groups[HDAWGChannelGrouping.CHAN_GROUP_4x2][2]

    @property
    def channel_pair_GH(self) -> 'HDAWGChannelGroup':
        return self._channel_groups[HDAWGChannelGrouping.CHAN_GROUP_4x2][3]

    @property
    def api_session(self) -> zhinst_core.ziDAQServer:
        return self._api_session

    @property
    def serial(self) -> str:
        return self._dev_ser

    def _initialize(self) -> None:
        settings = [(f'/{self.serial}/awgs/*/userregs/*', 0),  # Reset all user registers to 0.
                    (f'/{self.serial}/*/single', 1)]  # Single execution mode of sequence.
        for ch in range(0, 8):  # Route marker 1 signal for each channel to marker output.
            if ch % 2 == 0:
                output = HDAWGTriggerOutSource.OUT_1_MARK_1.value
            else:
                output = HDAWGTriggerOutSource.OUT_1_MARK_2.value
            settings.append(['/{}/triggers/out/{}/source'.format(self.serial, ch), output])

        self.api_session.set(settings)
        self.api_session.sync()  # Global sync: Ensure settings have taken effect on the device.

    def reset(self) -> None:
        zhinst.utils.disable_everything(self.api_session, self.serial)
        self._initialize()
        for tuple in self.channel_tuples:
            tuple.clear()
        self.api_session.set([
            (f'/{self.serial}/awgs/*/time', 0),
            (f'/{self.serial}/sigouts/*/range', HDAWGVoltageRange.RNG_1V.value),
            (f'/{self.serial}/awgs/*/outputs/*/amplitude', 1.0),
            (f'/{self.serial}/outputs/*/modulation/mode', HDAWGModulationMode.OFF.value),
        ])

        # marker outputs
        marker_settings = []
        for ch in range(0, 8):  # Route marker 1 signal for each channel to marker output.
            if ch % 2 == 0:
                output = HDAWGTriggerOutSource.OUT_1_MARK_1.value
            else:
                output = HDAWGTriggerOutSource.OUT_1_MARK_2.value
            marker_settings.append([f'/{self.serial}/triggers/out/{ch}/source', output])
        self.api_session.set(marker_settings)
        self.api_session.sync()

    def group_name(self, group_idx, group_size) -> str:
        if group_size is None:
            return f'{self.serial}_MDS'
        return str(self.serial) + '_' + 'ABCDEFGH'[group_idx*group_size:][:group_size]

    def _get_groups(self, grouping: 'HDAWGChannelGrouping') -> Tuple['HDAWGChannelGroup', ...]:
        try:
            return self._channel_groups[grouping]
        except KeyError:
            # python reload...
            for grouping_key, group in self._channel_groups.items():
                if grouping_key.value == grouping.value:
                    return group
            else:
                raise

    @property
    def channel_grouping(self) -> 'HDAWGChannelGrouping':
        grouping = self.api_session.getInt(f'/{self.serial}/SYSTEM/AWG/CHANNELGROUPING')
        return HDAWGChannelGrouping(grouping)

    @channel_grouping.setter
    def channel_grouping(self, channel_grouping: 'HDAWGChannelGrouping'):
        # ipython reload ...
        if not type(channel_grouping).__name__ == 'HDAWGChannelGrouping':
            raise HDAWGTypeError('Channel grouping must be an enum of type "HDAWGChannelGrouping" to avoid confusions '
                                 'between enum value and group size.')
        old_channel_grouping = self.channel_grouping
        if old_channel_grouping != channel_grouping:
            self.api_session.setInt(f'/{self.serial}/AWGS/*/ENABLE', 0)
            self.api_session.setInt(f'/{self.serial}/SYSTEM/AWG/CHANNELGROUPING', channel_grouping.value)
            # disable old groups
            for group in self._get_groups(old_channel_grouping):
                group.disconnect_group()

        if channel_grouping.value == HDAWGChannelGrouping.MDS.value and not self._is_mds_master():
            # do not connect channel group
            return

        for group in self._get_groups(channel_grouping):
            if not group.is_connected():
                group.connect_group(self)

    @valid_channel
    def offset(self, channel: int, voltage: float = None) -> float:
        """Query channel offset voltage and optionally set it."""
        return _sigout_offset(self.api_session, self.serial, channel, voltage)

    @valid_channel
    def range(self, channel: int, voltage: float = None) -> float:
        """Query channel voltage range and optionally set it. The instruments selects the next higher available range.
        This is the one-sided range Vp. Total range: -Vp...Vp"""
        return _sigout_range(self.api_session, self.serial, channel, voltage)

    @valid_channel
    def output(self, channel: int, status: bool = None) -> bool:
        """Query channel signal output status (enabled/disabled) and optionally set it. Corresponds to front LED."""
        return _sigout_on(self.api_session, self.serial, channel, status)

    def get_status_table(self):
        """Return node tree of instrument with all important settings, as well as each channel group as tuple."""
        return (self.api_session.get('/{}/*'.format(self.serial)),
                self.channel_pair_AB.awg_module.get('awgModule/*'),
                self.channel_pair_CD.awg_module.get('awgModule/*'),
                self.channel_pair_EF.awg_module.get('awgModule/*'),
                self.channel_pair_GH.awg_module.get('awgModule/*'))

    def _get_mds_group_idx(self) -> Optional[int]:
        idx = 0
        while True:
            try:
                if self.serial in self.api_session.getString(f'/ZI/MDS/GROUPS/{idx}/DEVICES'):
                    return idx
            except RuntimeError:
                break
            idx += 1

    def _is_mds_master(self) -> Optional[bool]:
        idx = 0
        while True:
            try:
                devices =  self.api_session.getString(f'/ZI/MDS/GROUPS/{idx}/DEVICES').split(',')
            except RuntimeError:
                break

            if self.serial in devices:
                return devices[0] == self.serial
            idx += 1

    def __repr__(self):
        return f"{type(self).__name__}({self.serial}, ... {self.api_session})"


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
    MDS = -1            # All channels that are in the current multi device synchronized group
    CHAN_GROUP_4x2 = 0  # 4x2 with HDAWG8; 2x2 with HDAWG4.  /dev.../awgs/0..3/
    CHAN_GROUP_2x4 = 1  # 2x4 with HDAWG8; 1x4 with HDAWG4.  /dev.../awgs/0 & 2/
    CHAN_GROUP_1x8 = 2  # 1x8 with HDAWG8.                   /dev.../awgs/0/

    def group_size(self) -> int:
        return {
            HDAWGChannelGrouping.CHAN_GROUP_4x2: 2,
            HDAWGChannelGrouping.CHAN_GROUP_2x4: 4,
            HDAWGChannelGrouping.CHAN_GROUP_1x8: 8,
            HDAWGChannelGrouping.MDS: None
        }[self]


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


@traced
class HDAWGChannelGroup(AWG):
    MIN_WAVEFORM_LEN = 192
    WAVEFORM_LEN_QUANTUM = 16

    def __init__(self,
                 identifier: str,
                 timeout: float) -> None:
        super().__init__(identifier)
        self.timeout = timeout

        self._awg_module = None
        self._program_manager = HDAWGProgramManager()
        self._elf_manager = None
        self._required_seqc_source = self._program_manager.to_seqc_program()
        self._uploaded_seqc_source = None
        self._current_program = None  # Currently armed program.
        self._upload_generator = ()

        self._master_device = None

    def _initialize_awg_module(self):
        """Only run once"""
        if self._awg_module:
            self._awg_module.clear()
        self._awg_module = self.master_device.api_session.awgModule()
        self._awg_module.set('awgModule/device', self.master_device.serial)
        self._awg_module.set('awgModule/index', self.awg_group_index)
        self._awg_module.execute()
        self._elf_manager = ELFManager(self._awg_module)
        self._upload_generator = ()

    @property
    def master_device(self) -> HDAWGRepresentation:
        """Reference to HDAWG representation."""
        if self._master_device is None:
            raise HDAWGValueError('Channel group is currently not connected')
        return self._master_device

    @property
    def awg_module(self) -> zhinst_core.AwgModule:
        """Each AWG channel group has its own awg module to manage program compilation and upload."""
        if self._awg_module is None:
            raise HDAWGValueError('Channel group is not connected and was never initialized')
        return self._awg_module

    @property
    @abstractmethod
    def awg_group_index(self) -> int:
        raise NotImplementedError()

    @property
    def num_markers(self) -> int:
        """Number of marker channels"""
        return 2 * self.num_channels

    def upload(self, name: str,
               program: Loop,
               channels: Tuple[Optional[ChannelID], ...],
               markers: Tuple[Optional[ChannelID], ...],
               voltage_transformation: Tuple[Callable, ...],
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
                        waveform_quantum=self.WAVEFORM_LEN_QUANTUM,
                        sample_rate=q_sample_rate)

        if self._amplitude_offset_handling == AWGAmplitudeOffsetHandling.IGNORE_OFFSET:
            voltage_offsets = (0.,) * self.num_channels
        elif self._amplitude_offset_handling == AWGAmplitudeOffsetHandling.CONSIDER_OFFSET:
            voltage_offsets = self.offsets()
        else:
            raise ValueError('{} is invalid as AWGAmplitudeOffsetHandling'.format(self._amplitude_offset_handling))

        amplitudes = self.amplitudes()

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
        self._program_manager.waveform_memory.sync_to_file_system(self.master_device.waveform_file_system)

        # start compiling the source (non-blocking)
        self._start_compile_and_upload()

    def _start_compile_and_upload(self):
        self._uploaded_seqc_source = None
        self._upload_generator = self._elf_manager.compile_and_upload(self._required_seqc_source)

    def _wait_for_compile_and_upload(self):
        for state in self._upload_generator:
            logger.debug("wait_for_compile_and_upload: %r", state)
            time.sleep(.1)
        self._uploaded_seqc_source = self._required_seqc_source
        logger.debug("AWG %d: wait_for_compile_and_upload has finished", self.awg_group_index)

    def was_current_program_finished(self) -> bool:
        """Return true if the current program has finished at least once"""
        playback_finished_mask = int(HDAWGProgramManager.Constants.PLAYBACK_FINISHED_MASK, 2)
        return bool(self.user_register(HDAWGProgramManager.Constants.PROG_SEL_REGISTER) & playback_finished_mask)

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
        self._start_compile_and_upload()
        self.arm(None)

    def arm(self, name: Optional[str]) -> None:
        """Load the program 'name' and arm the device for running it. If name is None the awg will "dearm" its current
        program.

        Currently hardware triggering is not implemented. The HDAWGProgramManager needs to emit code that calls
        `waitDigTrigger` to do that.
        """
        if self.num_channels > 8:
            if name is None:
                self._required_seqc_source = ""
            else:
                self._required_seqc_source = self._program_manager.to_seqc_program(name)
            self._start_compile_and_upload()

        if self._required_seqc_source != self._uploaded_seqc_source:
            self._wait_for_compile_and_upload()

        self.user_register(self._program_manager.Constants.TRIGGER_REGISTER, 0)

        if name is None:
            self.user_register(self._program_manager.Constants.PROG_SEL_REGISTER,
                               self._program_manager.Constants.PROG_SEL_NONE)
            self._current_program = None
        else:
            if name not in self.programs:
                raise HDAWGValueError('{} is unknown on {}'.format(name, self.identifier))
            self._current_program = name

            # set the registers of initial repetition counts
            for register, value in self._program_manager.get_register_values(name).items():
                assert register not in (self._program_manager.Constants.PROG_SEL_REGISTER,
                                        self._program_manager.Constants.TRIGGER_REGISTER)
                self.user_register(register, value)

            self.user_register(self._program_manager.Constants.PROG_SEL_REGISTER,
                               self._program_manager.name_to_index(name) | int(self._program_manager.Constants.NO_RESET_MASK, 2))

        # this was a workaround for problems in the past and I totally forgot why it was here
        # for ch_pair in self.master.channel_tuples:
        #    ch_pair._wait_for_compile_and_upload()
        self.enable(True)

    def run_current_program(self) -> None:
        """Run armed program."""
        if self._current_program is not None:
            if self._current_program not in self.programs:
                raise HDAWGValueError('{} is unknown on {}'.format(self._current_program, self.identifier))
            if not self.enable():
                self.enable(True)
            self.user_register(self._program_manager.Constants.TRIGGER_REGISTER,
                               int(self._program_manager.Constants.TRIGGER_RESET_MASK, 2))
        else:
            raise HDAWGRuntimeError('No program active')

    @property
    def programs(self) -> Set[str]:
        """The set of program names that can currently be executed on the hardware AWG."""
        return set(self._program_manager.programs.keys())

    @property
    def sample_rate(self) -> TimeType:
        """The default sample rate of the AWG channel group."""
        node_path = '/{}/awgs/{}/time'.format(self.master_device.serial, self.awg_group_index)
        sample_rate_num = self.master_device.api_session.getInt(node_path)
        node_path = '/{}/system/clocks/sampleclock/freq'.format(self.master_device.serial)
        sample_clock = self.master_device.api_session.getDouble(node_path)

        """Calculate exact rational number based on (sample_clock Sa/s) / 2^sample_rate_num. Otherwise numerical
        imprecision will give rise to errors for very long pulses. fractions.Fraction does not accept floating point
        numerator, which sample_clock could potentially be."""
        return time_from_float(sample_clock) / 2 ** sample_rate_num

    def connect_group(self, hdawg_device: HDAWGRepresentation):
        self.disconnect_group()
        self._master_device = weakref.proxy(hdawg_device)
        self._initialize_awg_module()
        # Seems creating AWG module sets SINGLE (single execution mode of sequence) to 0 per default.
        self.master_device.api_session.setInt(f'/{self.master_device.serial}/awgs/0/single', 1)

    def disconnect_group(self):
        """Disconnect this group from device so groups of another size can be used"""
        if self._awg_module:
            self.awg_module.clear()
        self._master_device = None
        self._elf_manager = None
        self._upload_generator = ()

    def is_connected(self) -> bool:
        return self._master_device is not None

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
            raise HDAWGValueError(f'{reg:!r} not a valid (1-16) register.')

        node_path = '/{}/awgs/{:d}/userregs/{:labone}'.format(self.master_device.serial, self.awg_group_index, reg)
        if value is not None:
            self.master_device.api_session.setInt(node_path, value)
            # hackedy
            for mds_serial in getattr(self, '_mds_devices', [])[1:]:
                self.master_device.api_session.setInt(node_path.replace(self.master_device.serial, mds_serial), value)
            self.master_device.api_session.sync()  # Global sync: Ensure settings have taken effect on the device.
        return self.master_device.api_session.getInt(node_path)


@traced
class MDSChannelGroup(HDAWGChannelGroup):
    def __init__(self,
                 identifier: str,
                 timeout: float) -> None:
        super().__init__(identifier, timeout)

        self._master_device = None
        self._mds_devices = None

    @property
    def num_channels(self) -> int:
        """Number of channels"""
        return len(self._mds_devices) * 8

    @property
    def awg_group_index(self):
        return 0

    def disconnect_group(self):
        super().disconnect_group()
        self._mds_devices = None

    def connect_group(self, hdawg_device: HDAWGRepresentation):
        mds_group = hdawg_device._get_mds_group_idx()
        if mds_group is None:
            raise HDAWGException("AWG not in any MDS group", hdawg_device)
        mds_devices = hdawg_device.api_session.getString(f'/ZI/MDS/GROUPS/{mds_group}/DEVICES').split(',')
        if hdawg_device.serial != mds_devices[0]:
            raise HDAWGException("Only the master device can connect to the HDAWG MDS channel group.")
        super().connect_group(hdawg_device)
        self._mds_devices = mds_devices

    def enable(self, status: bool = None) -> bool:
        """Start the AWG sequencer."""
        # There is also 'awgModule/awg/enable', which seems to have the same functionality.
        node_path = '/{}/awgs/{:d}/enable'.format(self.master_device.serial, 0)
        if status is not None:
            self.awg_module.set('awg/enable', int(status))
        else:
            status = self.awg_module.get('awg/module')

        #return bool(status)
        """
        if status is not None:
            self.master_device.api_session.setInt(node_path, int(status))
            for mds_device in self._mds_devices[1:]:
                self.master_device.api_session.setInt(node_path.replace(self._mds_devices[0], mds_device), int(status))
            self.master_device.api_session.sync()  # Global sync: Ensure settings have taken effect on the device.
            """
        return bool(self.master_device.api_session.getInt(node_path))

    def amplitudes(self) -> Tuple[float, ...]:
        """Query AWG channel amplitude value (not peak to peak).

        From manual:
        The final signal amplitude is given by the product of the full scale
        output range of 1 V[in this example], the dimensionless amplitude
        scaling factor 1.0, and the actual dimensionless signal amplitude
        stored in the waveform memory."""
        amplitudes = []

        api_session = self.master_device.api_session
        for mds_device in self._mds_devices:
            amplitude_scales = _amplitude_scales(api_session, mds_device)
            ranges = [_sigout_range(api_session, mds_device, ch) for ch in range(1, 9)]
            amplitudes.extend(zi_amplitude * zi_range / 2 for zi_amplitude, zi_range in zip(amplitude_scales, ranges))
        return tuple(amplitudes)

    def offsets(self) -> Tuple[float, ...]:
        offsets = []
        api_session = self.master_device.api_session
        for mds_device in self._mds_devices:
            offsets.extend(_sigout_offset(api_session, mds_device, ch) for ch in range(1, 9))
        return tuple(offsets)


class SingleDeviceChannelGroup(HDAWGChannelGroup):
    def __init__(self,
                 group_idx: int,
                 group_size: int,
                 identifier: str,
                 timeout: float) -> None:
        super().__init__(identifier, timeout)
        self._device = None

        assert group_idx in range(4)
        assert group_size in (2, 4, 8)

        self._group_idx = group_idx
        self._group_size = group_size

    @property
    def num_channels(self) -> int:
        """Number of channels"""
        return self._group_size

    def _channels(self, index_start=1) -> Tuple[int, ...]:
        """1 indexed channel"""
        offset = index_start + self._group_size * self._group_idx
        return tuple(ch + offset for ch in range(self.num_channels))

    @property
    def awg_group_index(self) -> int:
        """AWG node group index assuming 4x2 channel grouping. Then 0...3 will give appropriate index of group."""
        return self._group_idx

    @property
    def user_directory(self) -> str:
        """LabOne user directory with subdirectories: "awg/src" (seqc sourcefiles), "awg/elf" (compiled AWG binaries),
        "awag/waves" (user defined csv waveforms)."""
        return self.awg_module.getString('awgModule/directory')

    def enable(self, status: bool = None) -> bool:
        """Start the AWG sequencer."""
        # There is also 'awgModule/awg/enable', which seems to have the same functionality.
        node_path = '/{}/awgs/{:d}/enable'.format(self.master_device.serial, self.awg_group_index)
        if status is not None:
            self.master_device.api_session.setInt(node_path, int(status))
            self.master_device.api_session.sync()  # Global sync: Ensure settings have taken effect on the device.
        return bool(self.master_device.api_session.getInt(node_path))

    def amplitudes(self) -> Tuple[float, ...]:
        """Query AWG channel amplitude value (not peak to peak).

        From manual:
        The final signal amplitude is given by the product of the full scale
        output range of 1 V[in this example], the dimensionless amplitude
        scaling factor 1.0, and the actual dimensionless signal amplitude
        stored in the waveform memory."""
        amplitudes = []

        for ch, zi_amplitude in zip(self._channels(), _amplitude_scales(self.master_device.api_session, self.master_device.serial)):
            zi_range = self.master_device.range(ch)
            amplitudes.append(zi_amplitude * zi_range / 2)
        return tuple(amplitudes)

    def offsets(self) -> Tuple[float, ...]:
        return tuple(map(self.master_device.offset, self._channels()))


class ELFManager:
    class AWGModule:
        def __init__(self, awg_module: zhinst_core.AwgModule):
            """Provide an easily mockable interface to the zhinst AwgModule object"""
            self._module = awg_module

        @property
        def src_dir(self) -> pathlib.Path:
            return pathlib.Path(self._module.getString('directory'), 'awg', 'src')

        @property
        def elf_dir(self) -> pathlib.Path:
            return pathlib.Path(self._module.getString('directory'), 'awg', 'elf')

        @property
        def compiler_start(self) -> bool:
            """True if the compiler is running"""
            return self._module.getInt('compiler/start') == 1

        @compiler_start.setter
        def compiler_start(self, value: bool):
            """Set true to start the compiler"""
            self._module.set('compiler/start', value)

        @property
        def compiler_status(self) -> Tuple[int, str]:
            return self._module.getInt('compiler/status'), self._module.getString('compiler/statusstring')

        @property
        def compiler_source_file(self) -> str:
            return self._module.getString('compiler/sourcefile')

        @compiler_source_file.setter
        def compiler_source_file(self, source_file: str):
            self._module.set('compiler/sourcefile', source_file)

        @property
        def compiler_upload(self) -> bool:
            """auto upload after compiling"""
            return self._module.getInt('compiler/upload') == 1

        @compiler_upload.setter
        def compiler_upload(self, value: bool):
            self._module.set('compiler/upload', value)

        @property
        def elf_file(self) -> str:
            return self._module.getString('elf/file')

        @elf_file.setter
        def elf_file(self, elf_file: str):
            self._module.set('elf/file', elf_file)

        @property
        def elf_upload(self) -> bool:
            return bool(self._module.getInt('elf/upload'))

        @elf_upload.setter
        def elf_upload(self, value: bool):
            self._module.set('elf/upload', value)

        @property
        def elf_status(self) -> Tuple[int, float]:
            return self._module.getInt('elf/status'), self._module.getDouble('progress')

        @property
        def index(self) -> int:
            return self._module.getInt('index')

    def __init__(self, awg_module: zhinst_core.AwgModule):
        """This class organizes compiling and uploading of compiled programs. The source code file is named based on the
        code hash to cache compilation results. This requires that the waveform names are unique.

        The compilation and upload itself are done asynchronously by zhinst.core. To avoid spawning a useless
        thread for updating the status the method :py:meth:`~ELFManager.compile_and_upload` returns a generator which
        talks to the undelying library when needed."""
        self.awg_module = self.AWGModule(awg_module)

        # automatically upload after successful compilation
        self.awg_module.compiler_upload = True

        self._compile_job = None  # type: Optional[Union[str, Tuple[str, int, str]]]
        self._upload_job = None  # type: Optional[Union[Tuple[str, float], Tuple[str, int]]]

    def clear(self):
        """Deletes all files with a SHA512 hash name"""
        src_regex = re.compile(r'[a-z0-9]{128}\.seqc')
        elf_regex = re.compile(r'[a-z0-9]{128}\.elf')

        for p in self.awg_module.src_dir.iterdir():
            if src_regex.match(p.name):
                p.unlink()

        for p in self.awg_module.elf_dir.iterdir():
            if elf_regex.match(p.name):
                p.unlink()

    @staticmethod
    def _source_hash(source_string: str) -> str:
        """Calulate the SHA512 hash of the given source.

        Args:
            source_string: seqc source code

        Returns:
            hex representation of SHA512 `source_string` hash
        """
        # use utf-16 because str is UTF16 on most relevant machines (Windows)
        return hashlib.sha512(bytes(source_string, 'utf-16')).hexdigest()

    def _update_compile_job_status(self):
        """Store current compile status in self._compile_job."""
        compiler_start = self.awg_module.compiler_start
        if self._compile_job is None:
            assert compiler_start == 0

        elif isinstance(self._compile_job, str):
            if compiler_start:
                # compilation is running
                pass

            else:
                compiler_status, status_string = self.awg_module.compiler_status
                assert compiler_status in (-1, 0, 1, 2)
                if compiler_status == -1:
                    raise RuntimeError('Compile job is set but no compilation is running', status_string)
                elif compiler_status == 2:
                    logger.warning("AWG %d: Compilation finished with warning: %s", self.awg_module.index, status_string)
                self._compile_job = (self._compile_job, compiler_status, status_string)

    def _start_compile_job(self, source_file):
        logger.debug("Starting compilation of %r", source_file)
        self._update_compile_job_status()
        assert not isinstance(self._compile_job, str)
        self.awg_module.compiler_source_file = source_file
        self.awg_module.compiler_start = True
        self._compile_job = source_file
        logger.debug("AWG %d: Compilation of %r started", self.awg_module.index, source_file)

    def _compile(self, source_file) -> Generator[str, str, None]:
        self._start_compile_job(source_file)

        while True:
            self._update_compile_job_status()
            if not isinstance(self._compile_job, str):
                # finished compiling
                logger.debug("AWG %d: Compilation of %r finished", self.awg_module.index, source_file)
                break
            cmd = yield 'compiling'
            if cmd is None:
                logger.debug('No command received during compiling')
            elif cmd == 'abort':
                raise NotImplementedError('clean abort not implemented')
            else:
                raise HDAWGValueError('Unknown command', cmd)

        _, status_int, status_str = self._compile_job
        if status_int == 1:
            raise HDAWGRuntimeError('Compilation failed', status_str)
        logger.info("AWG %d: Compilation of %r successful", self.awg_module.index, source_file)

    def _start_elf_upload(self, elf_file):
        logger.debug("Uploading %r", elf_file)
        current_elf = self.awg_module.elf_file
        if current_elf != elf_file:
            logger.info("AWG %d: Overwriting elf file", self.awg_module.index)
            self.awg_module.elf_file = elf_file
            self.awg_module.elf_upload = True
        self._upload_job = (elf_file, None)
        time.sleep(.001)

    def _update_upload_job_status(self):
        elf_upload = self.awg_module.elf_upload
        if self._upload_job is None:
            assert not elf_upload
            return

        elf_file, old_status = self._upload_job
        assert self.awg_module.elf_file == elf_file

        if isinstance(old_status, float) or old_status is None:
            status_int, progress = self.awg_module.elf_status
            if status_int == 2:
                # in progress
                assert elf_upload == 1
                self._upload_job = elf_file, progress
            else:
                # fetch new value here
                self._upload_job = elf_file, status_int

        else:
            logger.debug('AWG %d: _update_upload_job_status called on finished upload', self.awg_module.index)
            assert elf_upload == 0

    def _upload(self, elf_file) -> Generator[str, str, None]:
        if self.awg_module.compiler_upload:
            pass
        else:
            self._start_elf_upload(elf_file)

        while True:
            self._update_upload_job_status()
            _, status = self._upload_job
            if isinstance(status, int):
                assert status in (-1, 0, 1)
                if status == 1:
                    raise RuntimeError('ELF upload failed')
                else:
                    break
            else:
                progress = status
                logger.debug('AWG %d: Upload progress is %d%%', self.awg_module.index, progress*100)

                cmd = yield 'uploading @ %d%%' % (100*progress)
                if cmd is None:
                    logger.debug("No command received during upload")
                if cmd == 'abort':
                    # TODO: check if this stops the upload
                    self.awg_module.elf_upload = False
                    raise NotImplementedError('Abort upload not cleanly implemented')
                else:
                    raise HDAWGValueError('Unknown command', cmd)

        # enable auto upload on compilation again
        # TODO: research whether this is necessary
        # self.awg_module.elf_file = ''

    def compile_and_upload(self, source_string: str) -> Generator[str, str, None]:
        """The source code is saved to a file determined by the source hash, compiled and uploaded to the instrument.
        The function returns a generator that yields the current state of the progress. The generator is empty iff the
        upload is complete. An exception is raised if there is an error.

        To abort send 'abort' to the generator.

        Example:
            >>> my_source = 'playWave("my_wave");'
            >>> for state in elf_manager.compile_and_upload(my_source):
            ...     print('Current state:', state)
            ...     time.sleep(1)

        Args:
            source_string: Source code to compile

        Returns:
            Generator object that needs to be consumed
        """
        self._update_compile_job_status()
        if isinstance(self._compile_job, str):
            raise NotImplementedError('cannot upload: compilation in progress')

        source_hash = self._source_hash(source_string)

        seqc_file_name = '%s.seqc' % source_hash
        elf_file_name = '%s.elf' % source_hash

        full_source_name = self.awg_module.src_dir.joinpath(seqc_file_name)
        full_elf_name = self.awg_module.elf_dir.joinpath(elf_file_name)

        if not full_source_name.exists():
            full_source_name.write_text(source_string, 'utf-8')

        # we assume same source == same program here
        if not full_elf_name.exists():
            yield from self._compile(seqc_file_name)
        else:
            # set this so the web interface shows the correct source
            # self.awg_module.compiler_source_file = seqc_file_name
            logger.info('Already compiled. ELF: %r', elf_file_name)

        yield from self._upload(elf_file_name)


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


def get_group_for_channels(hdawg: HDAWGRepresentation, channels: Set[int]) -> HDAWGChannelGroup:
    channels = set(channels)
    assert not channels - set(range(8)), "Channels must be in 0..=7"

    channel_range = range(min(channels) // 2 * 2, (max(channels) + 2) // 2 * 2)
    if len(channel_range) > 4 or len(channel_range) == 4 and channel_range.start == 2:
        c = (HDAWGChannelGrouping.CHAN_GROUP_1x8, 0)
    elif len(channel_range) == 4:
        assert channel_range.start in (0, 4)
        c = (HDAWGChannelGrouping.CHAN_GROUP_2x4, channel_range.start // 4)
    else:
        assert len(channel_range) == 2
        c = (HDAWGChannelGrouping.CHAN_GROUP_4x2, channel_range.start // 2)

    hdawg.channel_grouping = c[0]
    return hdawg.channel_tuples[c[1]]


def example_upload(hdawg_kwargs: dict, channels: Set[int], markers: Set[Tuple[int, int]]):  # pragma: no cover
    from qupulse.pulses import TablePT, SequencePT, RepetitionPT
    if isinstance(hdawg_kwargs, dict):
        hdawg = HDAWGRepresentation(**hdawg_kwargs)
    else:
        hdawg = hdawg_kwargs

    assert not set(channels) - set(range(8)), "Channels must be in 0..=7"
    channels = sorted(channels)

    required_channels = {*channels, *(ch for ch, _ in markers)}
    channel_group = get_group_for_channels(hdawg, required_channels)
    channel_group_channels = range(channel_group.awg_group_index * channel_group.num_channels,
                                   (channel_group.awg_group_index + 1) * channel_group.num_channels)

    # choose length based on minimal sample rate
    sample_rate = channel_group.sample_rate / 10**9
    min_t = channel_group.MIN_WAVEFORM_LEN / sample_rate
    quant_t = channel_group.WAVEFORM_LEN_QUANTUM / sample_rate

    assert min_t > 4 * quant_t, "Example not updated"

    entry_list1 = [(0, 0), (quant_t * 2, .2, 'hold'),    (min_t,  .3, 'linear'),   (min_t + 3*quant_t, 0, 'jump')]
    entry_list2 = [(0, 0), (quant_t * 3, -.2, 'hold'),   (min_t, -.3, 'linear'),  (min_t + 4*quant_t, 0, 'jump')]
    entry_list3 = [(0, 0), (quant_t * 1, -.2, 'linear'), (min_t, -.3, 'linear'), (2*min_t, 0, 'jump')]
    entry_lists = [entry_list1, entry_list2, entry_list3]

    entry_dict1 = {ch: entry_lists[:2][i % 2] for i, ch in enumerate(channels)}
    entry_dict2 = {ch: entry_lists[1::-1][i % 2] for i, ch in enumerate(channels)}
    entry_dict3 = {ch: entry_lists[2:0:-1][i % 2] for i, ch in enumerate(channels)}

    tpt1 = TablePT(entry_dict1, measurements=[('m', 20, 30)])
    tpt2 = TablePT(entry_dict2)
    tpt3 = TablePT(entry_dict3, measurements=[('m', 10, 50)])
    rpt = RepetitionPT(tpt1, 4)
    spt = SequencePT(tpt2, rpt)
    rpt2 = RepetitionPT(spt, 2)
    spt2 = SequencePT(rpt2, tpt3)
    p = spt2.create_program()

    upload_ch = tuple(ch if ch in channels else None
                      for ch in channel_group_channels)
    upload_mk = (None,) * channel_group.num_markers
    upload_vt = (lambda x: x,) * channel_group.num_channels

    channel_group.upload('pulse_test1', p, upload_ch, upload_mk, upload_vt)

    if markers:
        markers = sorted(markers)
        assert len(markers) == len(set(markers))
        channel_group_markers = tuple((ch, mk)
                                      for ch in channel_group_channels
                                      for mk in (0, 1))

        full_on = [(0, 1), (min_t, 1)]
        two_3rd = [(0, 1), (min_t*2/3, 0), (min_t, 0)]
        one_3rd = [(0, 0), (min_t*2/3, 1), (min_t, 1)]

        marker_start = TablePT({'m0': full_on, 'm1': full_on})
        marker_body = TablePT({'m0': two_3rd, 'm1': one_3rd})

        marker_test_pulse = marker_start @ RepetitionPT(marker_body, 10000)

        marker_program = marker_test_pulse.create_program()

        upload_ch = (None, ) * channel_group.num_channels
        upload_mk = tuple(f"m{mk}" if (ch, mk) in markers else None
                          for (ch, mk) in channel_group_markers)

        channel_group.upload('marker_test', marker_program, upload_ch, upload_mk, upload_vt)

    try:
        while True:
            for program in channel_group.programs:
                print(f'playing {program}')
                channel_group.arm(program)
                channel_group.run_current_program()
                while not channel_group.was_current_program_finished():
                    print(f'waiting for {program} to finish')
                    time.sleep(1e-2)
    finally:
        channel_group.enable(False)


if __name__ == "__main__":
    import sys
    args = argparse.ArgumentParser('Upload an example pulse to a HDAWG')
    args.add_argument('device_serial', help='device serial of the form dev1234')
    args.add_argument('device_interface', help='device interface', choices=['USB', '1GbE'], default='1GbE', nargs='?')
    args.add_argument('--channels', help='channels to use', choices=range(8), default=[0, 1], type=int, nargs='+')
    args.add_argument('--markers', help='markers to use', choices=range(8*2), default=[], type=int, nargs='*')
    parsed = vars(args.parse_args())

    channels = parsed.pop('channels')
    markers = [(m // 2, m % 2) for m in parsed.pop('markers')]

    logging.basicConfig(stream=sys.stdout)
    logger.setLevel(logging.DEBUG)
    example_upload(hdawg_kwargs=parsed, channels=channels, markers=markers)
