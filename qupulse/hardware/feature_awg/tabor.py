import functools
import logging
import numbers
import sys
import weakref
import warnings
from typing import List, Tuple, Set, Callable, Optional, Any, cast, Union, Dict, Mapping, NamedTuple, Iterable,\
    Collection, Sequence
from collections import OrderedDict

import numpy as np

from qupulse import ChannelID
from qupulse._program._loop import Loop, make_compatible

from qupulse.hardware.feature_awg.channel_tuple_wrapper import ChannelTupleAdapter
from qupulse.hardware.feature_awg.features import ChannelSynchronization, AmplitudeOffsetHandling, VoltageRange, \
    ProgramManagement, ActivatableChannels, DeviceControl, StatusTable, SCPI, VolatileParameters, \
    ReadProgram, RepetitionMode
from qupulse.hardware.util import voltage_to_uint16, find_positions

from qupulse.utils.types import TimeType
from qupulse.hardware.feature_awg.base import AWGChannelTuple, AWGChannel, AWGDevice, AWGMarkerChannel
from qupulse._program.tabor import TaborSegment, TaborException, TaborProgram, PlottableProgram, TaborSequencing, \
    make_combined_wave

import tabor_control.device
import pyvisa


assert (sys.byteorder == "little")

__all__ = ["TaborDevice", "TaborChannelTuple", "TaborChannel"]

TaborProgramMemory = NamedTuple("TaborProgramMemory", [("waveform_to_segment", np.ndarray),
                                                       ("program", TaborProgram)])


def with_configuration_guard(function_object: Callable[["TaborChannelTuple", Any], Any]) -> Callable[
    ["TaborChannelTuple"], Any]:
    """This decorator assures that the AWG is in configuration mode while the decorated method runs."""

    @functools.wraps(function_object)
    def guarding_method(channel_pair: "TaborChannelTuple", *args, **kwargs) -> Any:

        if channel_pair._configuration_guard_count == 0:
            channel_pair._enter_config_mode()
        channel_pair._configuration_guard_count += 1

        try:
            return function_object(channel_pair, *args, **kwargs)
        finally:
            channel_pair._configuration_guard_count -= 1
            if channel_pair._configuration_guard_count == 0:
                channel_pair._exit_config_mode()

    return guarding_method


def with_select(function_object: Callable[["TaborChannelTuple", Any], Any]) -> Callable[["TaborChannelTuple"], Any]:
    """Asserts the channel pair is selcted when the wrapped function is called"""

    @functools.wraps(function_object)
    def selector(channel_tuple: "TaborChannelTuple", *args, **kwargs) -> Any:
        channel_tuple._select()
        return function_object(channel_tuple, *args, **kwargs)

    return selector


########################################################################################################################
# Device
########################################################################################################################
# Features
class TaborSCPI(SCPI):
    def __init__(self, device: "TaborDevice", visa: pyvisa.resources.MessageBasedResource):
        super().__init__(visa)

        self._parent = weakref.ref(device)

    def send_cmd(self, cmd_str, paranoia_level=None):
        for instr in self._parent().all_devices:
            instr.send_cmd(cmd_str=cmd_str, paranoia_level=paranoia_level)

    def send_query(self, query_str, query_mirrors=False) -> Any:
        if query_mirrors:
            return tuple(instr.send_query(query_str) for instr in self._parent().all_devices)
        else:
            return self._parent().main_instrument.send_query(query_str)

    def _send_cmd(self, cmd_str, paranoia_level=None) -> Any:
        """Overwrite send_cmd for paranoia_level > 3"""
        if paranoia_level is None:
            paranoia_level = self._parent().paranoia_level

        if paranoia_level < 3:
            self._parent().super().send_cmd(cmd_str=cmd_str, paranoia_level=paranoia_level)  # pragma: no cover
        else:
            cmd_str = cmd_str.rstrip()

            if len(cmd_str) > 0:
                ask_str = cmd_str + "; *OPC?; :SYST:ERR?"
            else:
                ask_str = "*OPC?; :SYST:ERR?"

            *answers, opc, error_code_msg = self._parent()._visa_inst.ask(ask_str).split(";")

            error_code, error_msg = error_code_msg.split(",")
            error_code = int(error_code)
            if error_code != 0:
                _ = self._parent()._visa_inst.ask("*CLS; *OPC?")

                if error_code == -450:
                    # query queue overflow
                    self.send_cmd(cmd_str)
                else:
                    raise RuntimeError("Cannot execute command: {}\n{}: {}".format(cmd_str, error_code, error_msg))

            assert len(answers) == 0


class TaborChannelSynchronization(ChannelSynchronization):
    """This Feature is used to synchronise a certain ammount of channels"""

    def __init__(self, device: "TaborDevice"):
        super().__init__()
        self._parent = weakref.ref(device)

    def synchronize_channels(self, group_size: int) -> None:
        """
        Synchronize in groups of `group_size` channels. Groups of synchronized channels will be provided as
        AWGChannelTuples. The channel_size must be evenly dividable by the number of channels

        Args:
            group_size: Number of channels per channel tuple
        """
        if group_size == 2:
            self._parent()._channel_tuples = []
            for i in range((int)(len(self._parent().channels) / group_size)):
                self._parent()._channel_tuples.append(
                    TaborChannelTuple((i + 1),
                                      self._parent(),
                                      self._parent().channels[(i * group_size):((i * group_size) + group_size)],
                                      self._parent().marker_channels[(i * group_size):((i * group_size) + group_size)])
                )
            self._parent()[SCPI].send_cmd(":INST:COUP:STAT OFF")
        elif group_size == 4:
            self._parent()._channel_tuples = [TaborChannelTuple(1,
                                                                self._parent(),
                                                                self._parent().channels,
                                                                self._parent().marker_channels)]
            self._parent()[SCPI].send_cmd(":INST:COUP:STAT ON")
        else:
            raise TaborException("Invalid group size")


class TaborDeviceControl(DeviceControl):
    """This feature is used for basic communication with a AWG"""

    def __init__(self, device: "TaborDevice"):
        super().__init__()
        self._parent = weakref.ref(device)

    def reset(self) -> None:
        """
        Resetting the whole device. A command for resetting is send to the Device, the device is initialized again and
        all channel tuples are cleared.
        """
        self._parent()[SCPI].send_cmd(":RES")
        self._parent()._coupled = None

        self._parent()._initialize()
        for channel_tuple in self._parent().channel_tuples:
            channel_tuple[TaborProgramManagement].clear()

    def trigger(self) -> None:
        """
        This method triggers a device remotely.
        """
        self._parent()[SCPI].send_cmd(":TRIG")


class TaborStatusTable(StatusTable):
    def __init__(self, device: "TaborDevice"):
        super().__init__()
        self._parent = device

    def get_status_table(self) -> Dict[str, Union[str, float, int]]:
        """
        Send a lot of queries to the AWG about its settings. A good way to visualize is using pandas.DataFrame

        Returns:
            An ordered dictionary with the results
        """
        name_query_type_list = [("channel", ":INST:SEL?", int),
                                ("coupling", ":OUTP:COUP?", str),
                                ("volt_dc", ":SOUR:VOLT:LEV:AMPL:DC?", float),
                                ("volt_hv", ":VOLT:HV?", float),
                                ("offset", ":VOLT:OFFS?", float),
                                ("outp", ":OUTP?", str),
                                ("mode", ":SOUR:FUNC:MODE?", str),
                                ("shape", ":SOUR:FUNC:SHAPE?", str),
                                ("dc_offset", ":SOUR:DC?", float),
                                ("freq_rast", ":FREQ:RAST?", float),

                                ("gated", ":INIT:GATE?", str),
                                ("continuous", ":INIT:CONT?", str),
                                ("continuous_enable", ":INIT:CONT:ENAB?", str),
                                ("continuous_source", ":INIT:CONT:ENAB:SOUR?", str),
                                ("marker_source", ":SOUR:MARK:SOUR?", str),
                                ("seq_jump_event", ":SOUR:SEQ:JUMP:EVEN?", str),
                                ("seq_adv_mode", ":SOUR:SEQ:ADV?", str),
                                ("aseq_adv_mode", ":SOUR:ASEQ:ADV?", str),

                                ("marker", ":SOUR:MARK:SEL?", int),
                                ("marker_high", ":MARK:VOLT:HIGH?", str),
                                ("marker_low", ":MARK:VOLT:LOW?", str),
                                ("marker_width", ":MARK:WIDT?", int),
                                ("marker_state", ":MARK:STAT?", str)]

        data = OrderedDict((name, []) for name, *_ in name_query_type_list)
        for ch in (1, 2, 3, 4):
            self._parent.channels[ch - 1]._select()
            self._parent.marker_channels[(ch - 1) % 2]._select()
            for name, query, dtype in name_query_type_list:
                data[name].append(dtype(self._parent[SCPI].send_query(query)))
        return data


# Implementation
class TaborDevice(AWGDevice):
    def __init__(self, device_name: str, instr_addr=None, paranoia_level=1, external_trigger=False, reset=False,
                 mirror_addresses=()):
        """
        Constructor for a Tabor device

        Args:
            device_name (str):       Name of the device
            instr_addr:              Instrument address that is forwarded to tabor_control
            paranoia_level (int):    Paranoia level that is forwarded to tabor_control
            external_trigger (bool): Not supported yet
            reset (bool):
            mirror_addresses:        list of devices on which the same things as on the main device are done.
                                     For example you can a simulator and a real Device at once
        """
        super().__init__(device_name)
        self._instr = tabor_control.device.TEWXAwg(tabor_control.open_session(instr_addr), paranoia_level)
        self._mirrors = tuple(tabor_control.device.TEWXAwg(tabor_control.open_session(address), paranoia_level)
                              for address in mirror_addresses)
        self._coupled = None
        self._clock_marker = [0, 0, 0, 0]

        self.add_feature(TaborSCPI(self, self.main_instrument._visa_inst))
        self.add_feature(TaborDeviceControl(self))
        self.add_feature(TaborStatusTable(self))

        if reset:
            self[SCPI].send_cmd(":RES")

        # Channel
        self._channels = [TaborChannel(i + 1, self) for i in range(4)]

        # MarkerChannels
        self._marker_channels = [TaborMarkerChannel(i + 1, self) for i in range(4)]

        self._initialize()

        # ChannelTuple
        self._channel_tuples = []

        self.add_feature(TaborChannelSynchronization(self))
        self[TaborChannelSynchronization].synchronize_channels(2)

        if external_trigger:
            raise NotImplementedError()  # pragma: no cover

    def enable(self) -> None:
        """
        This method immediately generates the selected output waveform, if the device is in continuous and armed
        repetition mode.
        """
        self[SCPI].send_cmd(":ENAB")

    def abort(self) -> None:
        """
        With abort you can terminate the current generation of the output waveform. When the output waveform is
        terminated the output starts generating an idle waveform.
        """
        self[SCPI].send_cmd(":ABOR")

    def set_coupled(self, coupled: bool) -> None:
        """
        Thats the coupling of the device to 'coupled'
        """
        if coupled:
            self[SCPI].send_cmd("INST:COUP:STAT ON")
        else:
            self[SCPI].send_cmd("INST:COUP:STAT OFF")

    def _is_coupled(self) -> bool:
        """
        Returns true if the coupling of the device is 'coupled' otherwise false
        """
        if self._coupled is None:
            return self[SCPI].send_query(":INST:COUP:STAT?") == "ON"
        else:
            return self._coupled

    def cleanup(self) -> None:
        for channel_tuple in self.channel_tuples:
            channel_tuple.cleanup()

    @property
    def channels(self) -> Collection["TaborChannel"]:
        """Returns a list of all channels of a Device"""
        return self._channels

    @property
    def marker_channels(self) -> Collection["TaborMarkerChannel"]:
        """Returns a list of all marker channels of a device. The collection may be empty"""
        return self._marker_channels

    @property
    def channel_tuples(self) -> Collection["TaborChannelTuple"]:
        """Returns a list of all channel tuples of a list"""
        return self._channel_tuples

    @property
    def main_instrument(self) -> tabor_control.device.TEWXAwg:
        return self._instr

    @property
    def mirrored_instruments(self) -> Sequence[tabor_control.device.TEWXAwg]:
        return self._mirrors

    @property
    def all_devices(self) -> Sequence[tabor_control.device.TEWXAwg]:
        return (self._instr,) + self._mirrors

    @property
    def _paranoia_level(self) -> tabor_control.ParanoiaLevel:
        return self._instr.paranoia_level

    @_paranoia_level.setter
    def _paranoia_level(self, val):
        for instr in self.all_devices:
            instr.paranoia_level = val

    @property
    def dev_properties(self) -> dict:
        return self._instr.dev_properties.as_dict()

    def _send_binary_data(self, bin_dat, paranoia_level=None):
        for instr in self.all_devices:
            instr.write_segment_data(bin_dat, paranoia_level=paranoia_level)

    def _download_segment_lengths(self, seg_len_list, paranoia_level=None):
        for instr in self.all_devices:
            instr.write_segment_lengths(seg_len_list, paranoia_level=paranoia_level)

    def _download_sequencer_table(self, seq_table, paranoia_level=None):
        for instr in self.all_devices:
            instr.write_sequencer_table(seq_table, paranoia_level=paranoia_level)

    def _download_adv_seq_table(self, seq_table, paranoia_level=None):
        for instr in self.all_devices:
            instr.write_advanced_sequencer_table(seq_table, paranoia_level=paranoia_level)

    def _initialize(self) -> None:
        # 1. Select channel
        # 2. Turn off gated mode
        # 3. Turn on continous mode
        # 4. Armed mode (only generate waveforms after enab command)
        # 5. Expect enable signal from (USB / LAN / GPIB)
        # 6. Use arbitrary waveforms as marker source
        # 7. Expect jump command for sequencing from (USB / LAN / GPIB)

        setup_command = (
            ":INIT:GATE OFF; :INIT:CONT ON; "
            ":INIT:CONT:ENAB ARM; :INIT:CONT:ENAB:SOUR BUS;"
            ":SOUR:MARK:SOUR USER; :SOUR:SEQ:JUMP:EVEN BUS ")
        self[SCPI].send_cmd(":INST:SEL 1")
        self[SCPI].send_cmd(setup_command)
        self[SCPI].send_cmd(":INST:SEL 3")
        self[SCPI].send_cmd(setup_command)

    def _get_readable_device(self, simulator=True) -> tabor_control.device.TEWXAwg:
        """
        A method to get the first readable device out of all devices.
        A readable device is a device which you can read data from like a simulator.

        Returns:
            The first readable device out of all devices

        Throws:
            TaborException: this exception is thrown if there is no readable device in the list of all devices
        """
        for device in self.all_devices:
            if device.supports_basic_reading():
                if simulator:
                    if device.is_simulator:
                        return device
                else:
                    return device
        raise TaborException("No device capable of device data read")


########################################################################################################################
# Channel
########################################################################################################################
# Features
class TaborVoltageRange(VoltageRange):
    def __init__(self, channel: "TaborChannel"):
        super().__init__()
        self._parent = weakref.ref(channel)

    @property
    @with_select
    def offset(self) -> float:
        """Get offset of AWG channel"""
        return float(
            self._parent().device[SCPI].send_query(":VOLT:OFFS?".format(channel=self._parent().idn)))

    @property
    @with_select
    def amplitude(self) -> float:
        """Get amplitude of AWG channel"""
        coupling = self._parent().device[SCPI].send_query(":OUTP:COUP?")
        if coupling == "DC":
            return float(self._parent().device[SCPI].send_query(":VOLT?"))
        elif coupling == "HV":
            return float(self._parent().device[SCPI].send_query(":VOLT:HV?"))
        else:
            raise TaborException("Unknown coupling: {}".format(coupling))

    @property
    def amplitude_offset_handling(self) -> AmplitudeOffsetHandling:
        """
        Gets the amplitude and offset handling of this channel. The amplitude-offset controls if the amplitude and
        offset settings are constant or if these should be optimized by the driver
        """
        return self._parent()._amplitude_offset_handling

    @amplitude_offset_handling.setter
    def amplitude_offset_handling(self, amp_offs_handling: Union[AmplitudeOffsetHandling, str]) -> None:
        """
        amp_offs_handling: See possible values at `AWGAmplitudeOffsetHandling`
        """
        amp_offs_handling = AmplitudeOffsetHandling(AmplitudeOffsetHandling)
        self._parent()._amplitude_offset_handling = amp_offs_handling

    def _select(self) -> None:
        self._parent()._select()


class TaborActivatableChannels(ActivatableChannels):
    def __init__(self, channel: "TaborChannel"):
        super().__init__()
        self._parent = weakref.ref(channel)

    @property
    def enabled(self) -> bool:
        """
        Returns the the state a channel has at the moment. A channel is either activated or deactivated
        True stands for activated and false for deactivated
        """
        return self._parent().device[SCPI].send_query(":OUTP ?") == "ON"

    @with_select
    def enable(self):
        """Enables the output of a certain channel"""
        command_string = ":OUTP ON".format(ch_id=self._parent().idn)
        self._parent().device[SCPI].send_cmd(command_string)

    @with_select
    def disable(self):
        """Disables the output of a certain channel"""
        command_string = ":OUTP OFF".format(ch_id=self._parent().idn)
        self._parent().device[SCPI].send_cmd(command_string)

    def _select(self) -> None:
        self._parent()._select()

# Implementation
class TaborChannel(AWGChannel):
    def __init__(self, idn: int, device: TaborDevice):
        super().__init__(idn)

        self._device = weakref.ref(device)
        self._amplitude_offset_handling = AmplitudeOffsetHandling.IGNORE_OFFSET

        # adding Features
        self.add_feature(TaborVoltageRange(self))
        self.add_feature(TaborActivatableChannels(self))

    @property
    def device(self) -> TaborDevice:
        """Returns the device that the channel belongs to"""
        return self._device()

    @property
    def channel_tuple(self) -> "TaborChannelTuple":
        """Returns the channel tuple that this channel belongs to"""
        return self._channel_tuple()

    def _set_channel_tuple(self, channel_tuple: "TaborChannelTuple") -> None:
        """
        The channel tuple "channel_tuple" is assigned to this channel

        Args:
            channel_tuple (TaborChannelTuple): the channel tuple that this channel belongs to
        """
        self._channel_tuple = weakref.ref(channel_tuple)

    def _select(self) -> None:
        self.device[SCPI].send_cmd(":INST:SEL {channel}".format(channel=self.idn))


########################################################################################################################
# ChannelTuple
########################################################################################################################
# Features
class TaborProgramManagement(ProgramManagement):
    def __init__(self, channel_tuple: "TaborChannelTuple"):
        super().__init__(channel_tuple)
        self._programs = {}
        self._armed_program = None

        self._idle_sequence_table = [(1, 1, 0), (1, 1, 0), (1, 1, 0)]
        self._trigger_source = 'BUS'

    def get_repetition_mode(self, program_name: str) -> str:
        """
        Returns the default repetition mode of a certain program
        Args:
            program_name (str): name of the program whose repetition mode should be returned
        """
        return self._channel_tuple._known_programs[program_name].program._repetition_mode

    def set_repetition_mode(self, program_name: str, repetition_mode: str) -> None:
        """
        Changes the default repetition mode of a certain program

        Args:
            program_name (str): name of the program whose repetition mode should be changed

        Throws:
            ValueError: this Exception is thrown when an invalid repetition mode is given
        """
        if repetition_mode in ("infinite", "once"):
            self._channel_tuple._known_programs[program_name].program._repetition_mode = repetition_mode
        else:
            raise ValueError("{} is no vaild repetition mode".format(repetition_mode))

    @property
    def supported_repetition_modes(self) -> Set[RepetitionMode]:
        return {RepetitionMode.INFINITE}

    @with_configuration_guard
    @with_select
    def upload(self, name: str,
               program: Loop,
               channels: Tuple[Optional[ChannelID], Optional[ChannelID]],
               marker_channels: Tuple[Optional[ChannelID], Optional[ChannelID]],
               voltage_transformation: Tuple[Callable, Callable],
               repetition_mode: str = None,
               force: bool = False) -> None:
        """
        Upload a program to the AWG.

        The policy is to prefer amending the unknown waveforms to overwriting old ones.
        """

        if repetition_mode is None:
            repetition_mode = self._default_repetition_mode
        else:
            repetition_mode = RepetitionMode(repetition_mode)

        if repetition_mode not in self.supported_repetition_modes:
            raise ValueError(f"{repetition_mode} is not supported on {self._channel_tuple}")
        if len(channels) != len(self._channel_tuple.channels):
            raise ValueError("Wrong number of channels")
        if len(marker_channels) != len(self._channel_tuple.marker_channels):
            raise ValueError("Wrong number of marker")
        if len(voltage_transformation) != len(self._channel_tuple.channels):
            raise ValueError("Wrong number of voltage transformations")

        # adjust program to fit criteria
        sample_rate = self._channel_tuple.device.channel_tuples[0].sample_rate
        make_compatible(program,
                        minimal_waveform_length=192,
                        waveform_quantum=16,
                        sample_rate=sample_rate / 10 ** 9)

        if name in self._channel_tuple._known_programs:
            if force:
                self._channel_tuple.free_program(name)
            else:
                raise ValueError('{} is already known on {}'.format(name, self._channel_tuple.idn))

        # They call the peak to peak range amplitude

        ranges = tuple(ch[VoltageRange].amplitude for ch in self._channel_tuple.channels)

        voltage_amplitudes = tuple(range / 2 for range in ranges)

        voltage_offsets = []
        for channel in self._channel_tuple.channels:
            if channel._amplitude_offset_handling == AmplitudeOffsetHandling.IGNORE_OFFSET:
                voltage_offsets.append(0)
            elif channel._amplitude_offset_handling == AmplitudeOffsetHandling.CONSIDER_OFFSET:
                voltage_offsets.append(channel[VoltageRange].offset)
            else:
                raise NotImplementedError(
                    '{} is invalid as AWGAmplitudeOffsetHandling'.format(channel._amplitude_offset_handling))
        voltage_offsets = tuple(voltage_offsets)

        # parse to tabor program
        tabor_program = TaborProgram(program,
                                     channels=tuple(channels),
                                     markers=marker_channels,
                                     device_properties=self._channel_tuple.device.dev_properties,
                                     sample_rate=sample_rate / 10 ** 9,
                                     amplitudes=voltage_amplitudes,
                                     offsets=voltage_offsets,
                                     voltage_transformations=voltage_transformation)

        segments, segment_lengths = tabor_program.get_sampled_segments()

        waveform_to_segment, to_amend, to_insert = self._channel_tuple._find_place_for_segments_in_memory(segments,
                                                                                                          segment_lengths)

        self._channel_tuple._segment_references[waveform_to_segment[waveform_to_segment >= 0]] += 1

        for wf_index in np.flatnonzero(to_insert > 0):
            segment_index = to_insert[wf_index]
            self._channel_tuple._upload_segment(to_insert[wf_index], segments[wf_index])
            waveform_to_segment[wf_index] = segment_index

        if np.any(to_amend):
            segments_to_amend = [segments[idx] for idx in np.flatnonzero(to_amend)]
            waveform_to_segment[to_amend] = self._channel_tuple._amend_segments(segments_to_amend)

        self._channel_tuple._known_programs[name] = TaborProgramMemory(waveform_to_segment=waveform_to_segment,
                                                                         program=tabor_program)

        # set the default repetionmode for a programm
        self.set_repetition_mode(program_name=name, repetition_mode=repetition_mode)

    def remove(self, name: str) -> None:
        """
        Remove a program from the AWG.

        Also discards all waveforms referenced only by the program identified by name.

        Args:
            name (str): The name of the program to remove.
        """
        self._channel_tuple.free_program(name)
        self._channel_tuple.cleanup()

    def clear(self) -> None:
        """
        Removes all programs and waveforms from the AWG.

        Caution: This affects all programs and waveforms on the AWG, not only those uploaded using qupulse!
        """

        self._channel_tuple.device.channels[0]._select()
        self._channel_tuple.device[SCPI].send_cmd(":TRAC:DEL:ALL")
        self._channel_tuple.device[SCPI].send_cmd(":SOUR:SEQ:DEL:ALL")
        self._channel_tuple.device[SCPI].send_cmd(":ASEQ:DEL")

        self._channel_tuple.device[SCPI].send_cmd(":TRAC:DEF 1, 192")
        self._channel_tuple.device[SCPI].send_cmd(":TRAC:SEL 1")
        self._channel_tuple.device[SCPI].send_cmd(":TRAC:MODE COMB")
        self._channel_tuple.device._send_binary_data(bin_dat=self._channel_tuple._idle_segment.get_as_binary())

        self._channel_tuple._segment_lengths = 192 * np.ones(1, dtype=np.uint32)
        self._channel_tuple._segment_capacity = 192 * np.ones(1, dtype=np.uint32)
        self._channel_tuple._segment_hashes = np.ones(1, dtype=np.int64) * hash(self._channel_tuple._idle_segment)
        self._channel_tuple._segment_references = np.ones(1, dtype=np.uint32)

        self._channel_tuple._advanced_sequence_table = []
        self._channel_tuple._sequencer_tables = []

        self._channel_tuple._known_programs = dict()
        self._change_armed_program(None)

    @with_select
    def arm(self, name: Optional[str]) -> None:
        """
        Load the program 'name' and arm the device for running it.

        Args:
            name (str): the program the device should change to
        """
        if self._channel_tuple._current_program == name:
            self._channel_tuple.device[SCPI].send_cmd("SEQ:SEL 1")
        else:
            self._change_armed_program(name)

    @property
    def programs(self) -> Set[str]:
        """The set of program names that can currently be executed on the hardware AWG."""
        return set(program.name for program in self._channel_tuple._known_programs.keys())

    @with_select
    def run_current_program(self) -> None:
        """
        This method starts running the active program

        Throws:
            RuntimeError: This exception is thrown if there is no active program for this device
        """
        if (self._channel_tuple.device._is_coupled()):
            # channel tuple is the first channel tuple
            if (self._channel_tuple.device._channel_tuples[0] == self):
                if self._channel_tuple._current_program:
                    repetition_mode = self._channel_tuple._known_programs[
                        self._channel_tuple._current_program].program._repetition_mode
                    if repetition_mode == "infinite":
                        self._cont_repetition_mode()
                        self._channel_tuple.device[SCPI].send_cmd(':TRIG',
                                                                    paranoia_level=self._channel_tuple.internal_paranoia_level)
                    else:
                        raise ValueError("{} is no vaild repetition mode".format(repetition_mode))
                else:
                    raise RuntimeError("No program active")
            else:
                warnings.warn(
                    "TaborWarning - run_current_program() - the device is coupled - runthe program via the first channel tuple")

        else:
            if self._channel_tuple._current_program:
                repetition_mode = self._channel_tuple._known_programs[
                    self._channel_tuple._current_program].program._repetition_mode
                if repetition_mode == "infinite":
                    self._cont_repetition_mode()
                    self._channel_tuple.device[SCPI].send_cmd(':TRIG', paranoia_level=self._channel_tuple.internal_paranoia_level)
                else:
                    raise ValueError("{} is no vaild repetition mode".format(repetition_mode))
            else:
                raise RuntimeError("No program active")

    @with_select
    @with_configuration_guard
    def _change_armed_program(self, name: Optional[str]) -> None:
        """The armed program of the channel tuple is changed to the program with the name 'name'"""
        if name is None:
            sequencer_tables = [self._idle_sequence_table]
            advanced_sequencer_table = [(1, 1, 0)]
        else:
            waveform_to_segment_index, program = self._channel_tuple._known_programs[name]
            waveform_to_segment_number = waveform_to_segment_index + 1

            # translate waveform number to actual segment
            sequencer_tables = [[(rep_count, waveform_to_segment_number[wf_index], jump_flag)
                                 for ((rep_count, wf_index, jump_flag), _) in sequencer_table]
                                for sequencer_table in program.get_sequencer_tables()]

            # insert idle sequence
            sequencer_tables = [self._idle_sequence_table] + sequencer_tables

            # adjust advanced sequence table entries by idle sequence table offset
            advanced_sequencer_table = [(rep_count, seq_no + 1, jump_flag)
                                        for rep_count, seq_no, jump_flag in program.get_advanced_sequencer_table()]

            if program.waveform_mode == TaborSequencing.SINGLE:
                assert len(advanced_sequencer_table) == 1
                assert len(sequencer_tables) == 2

                while len(sequencer_tables[1]) < self._channel_tuple.device.dev_properties["min_seq_len"]:
                    assert advanced_sequencer_table[0][0] == 1
                    sequencer_tables[1].append((1, 1, 0))

        # insert idle sequence in advanced sequence table
        advanced_sequencer_table = [(1, 1, 0)] + advanced_sequencer_table

        while len(advanced_sequencer_table) < self._channel_tuple.device.dev_properties["min_aseq_len"]:
            advanced_sequencer_table.append((1, 1, 0))

        self._channel_tuple.device[SCPI].send_cmd("SEQ:DEL:ALL", paranoia_level=self._channel_tuple.internal_paranoia_level)
        self._channel_tuple._sequencer_tables = []
        self._channel_tuple.device[SCPI].send_cmd("ASEQ:DEL", paranoia_level=self._channel_tuple.internal_paranoia_level)
        self._channel_tuple._advanced_sequence_table = []

        # download all sequence tables
        for i, sequencer_table in enumerate(sequencer_tables):
            self._channel_tuple.device[SCPI].send_cmd("SEQ:SEL {}".format(i + 1),
                                                        paranoia_level=self._channel_tuple.internal_paranoia_level)
            self._channel_tuple.device._download_sequencer_table(sequencer_table)
        self._channel_tuple._sequencer_tables = sequencer_tables
        self._channel_tuple.device[SCPI].send_cmd("SEQ:SEL 1", paranoia_level=self._channel_tuple.internal_paranoia_level)

        self._channel_tuple.device._download_adv_seq_table(advanced_sequencer_table)
        self._channel_tuple._advanced_sequence_table = advanced_sequencer_table

        self._channel_tuple._current_program = name

    def _select(self):
        self._channel_tuple.channels[0]._select()

    @property
    def _configuration_guard_count(self):
        return self._channel_tuple._configuration_guard_count

    @_configuration_guard_count.setter
    def _configuration_guard_count(self, configuration_guard_count):
        self._channel_tuple._configuration_guard_count = configuration_guard_count

    def _enter_config_mode(self):
        self._channel_tuple._enter_config_mode()

    def _exit_config_mode(self):
        self._channel_tuple._exit_config_mode()

    @with_select
    def _cont_repetition_mode(self):
        """Changes the run mode of this channel tuple to continous mode"""
        self._channel_tuple.device[SCPI].send_cmd(f":TRIG:SOUR:ADV EXT")
        self._channel_tuple.device[SCPI].send_cmd(
            f":INIT:GATE OFF; :INIT:CONT ON; :INIT:CONT:ENAB ARM; :INIT:CONT:ENAB:SOUR {self._trigger_source}")


class TaborVolatileParameters(VolatileParameters):
    def __init__(self, channel_tuple: "TaborChannelTuple", ):
        super().__init__(channel_tuple=channel_tuple)

    def set_volatile_parameters(self, program_name: str, parameters: Mapping[str, numbers.Number]) -> None:
        """ Set the values of parameters which were marked as volatile on program creation. Sets volatile parameters
        in program memory and device's (adv.) sequence tables if program is current program.

        If set_volatile_parameters needs to run faster, set CONFIG_MODE_PARANOIA_LEVEL to 0 which causes the device to
        enter the configuration mode with paranoia level 0 (Note: paranoia level 0 does not work for the simulator)
        and set device._is_coupled.

        Args:
            program_name: Name of program which should be changed.
            parameters: Names of volatile parameters and respective values to which they should be set.
        """
        waveform_to_segment_index, program = self._channel_tuple._known_programs[program_name]
        modifications = program.update_volatile_parameters(parameters)

        self._channel_tuple.logger.debug("parameter modifications: %r" % modifications)

        if not modifications:
            self._channel_tuple.logger.info(
                "There are no volatile parameters to update. Either there are no volatile parameters with "
                "these names,\nthe respective repetition counts already have the given values or the "
                "volatile parameters were dropped during upload.")
            return

        if program_name == self._channel_tuple._current_program:
            commands = []

            for position, entry in modifications.items():
                if not entry.repetition_count > 0:
                    raise ValueError("Repetition must be > 0")

                if isinstance(position, int):
                    commands.append(":ASEQ:DEF {},{},{},{}".format(position + 1, entry.element_number + 1,
                                                                   entry.repetition_count, entry.jump_flag))
                else:
                    table_num, step_num = position
                    commands.append(":SEQ:SEL {}".format(table_num + 2))
                    commands.append(":SEQ:DEF {},{},{},{}".format(step_num,
                                                                  waveform_to_segment_index[entry.element_id] + 1,
                                                                  entry.repetition_count, entry.jump_flag))
            self._channel_tuple._execute_multiple_commands_with_config_guard(commands)

        # Wait until AWG is finished
        _ = self._channel_tuple.device.main_instrument._visa_inst.query("*OPC?")


class TaborReadProgram(ReadProgram):
    def __init__(self, channel_tuple: "TaborChannelTuple", ):
        super().__init__(channel_tuple=channel_tuple)

    def read_complete_program(self):
        return PlottableProgram.from_read_data(self._channel_tuple.read_waveforms(),
                                               self._channel_tuple.read_sequence_tables(),
                                               self._channel_tuple.read_advanced_sequencer_table())


# Implementation
class TaborChannelTuple(AWGChannelTuple):
    CONFIG_MODE_PARANOIA_LEVEL = None

    def __init__(self, idn: int, device: TaborDevice, channels: Iterable["TaborChannel"],
                 marker_channels: Iterable["TaborMarkerChannel"]):
        super().__init__(idn)
        self._device = weakref.ref(device)

        self._configuration_guard_count = 0
        self._is_in_config_mode = False

        self._channels = tuple(channels)
        self._marker_channels = tuple(marker_channels)

        # the channel and channel marker are assigned to this channel tuple
        for channel in self.channels:
            channel._set_channel_tuple(self)
        for marker_ch in self.marker_channels:
            marker_ch._set_channel_tuple(self)

        # adding Features
        self.add_feature(TaborProgramManagement(self))
        self.add_feature(TaborVolatileParameters(self))

        self._idle_segment = TaborSegment.from_sampled(voltage_to_uint16(voltage=np.zeros(192),
                                                                         output_amplitude=0.5,
                                                                         output_offset=0., resolution=14),
                                                       voltage_to_uint16(voltage=np.zeros(192),
                                                                         output_amplitude=0.5,
                                                                         output_offset=0., resolution=14),
                                                       None, None)

        self._known_programs = dict()  # type: Dict[str, TaborProgramMemory]
        self._current_program = None

        self._segment_lengths = None
        self._segment_capacity = None
        self._segment_hashes = None
        self._segment_references = None

        self._sequencer_tables = None
        self._advanced_sequence_table = None

        self._internal_paranoia_level = 0

        self[TaborProgramManagement].clear()

        self._channel_tuple_adapter: ChannelTupleAdapter

    @property
    def internal_paranoia_level(self) -> Optional[int]:
        return self._internal_paranoia_level

    @property
    def logger(self):
        return logging.getLogger("qupulse.tabor")

    @property
    def channel_tuple_adapter(self) -> ChannelTupleAdapter:
        if self._channel_tuple_adapter is None:
            self._channel_tuple_adapter = ChannelTupleAdapter(self)
        return self._channel_tuple_adapter

    def _select(self) -> None:
        """The channel tuple is selected, which means that the first channel of the channel tuple is selected"""
        self.channels[0]._select()

    @property
    def device(self) -> TaborDevice:
        """Returns the device that the channel tuple belongs to"""
        return self._device()

    @property
    def channels(self) -> Collection["TaborChannel"]:
        """Returns all channels of the channel tuple"""
        return self._channels

    @property
    def marker_channels(self) -> Collection["TaborMarkerChannel"]:
        """Returns all marker channels of the channel tuple"""
        return self._marker_channels

    @property
    @with_select
    def sample_rate(self) -> TimeType:
        """Returns the sample rate that the channels of a channel tuple have"""
        return TimeType.from_float(
            float(self.device[SCPI].send_query(":FREQ:RAST?".format(channel=self.channels[0].idn))))

    @property
    def total_capacity(self) -> int:
        return int(self.device.dev_properties["max_arb_mem"]) // 2

    def free_program(self, name: str) -> TaborProgramMemory:
        if name is None:
            raise TaborException("Removing 'None' program is forbidden.")
        program = self._known_programs.pop(name)
        self._segment_references[program.waveform_to_segment] -= 1
        if self._current_program == name:
            self[TaborProgramManagement]._change_armed_program(None)
        return program

    @property
    def _segment_reserved(self) -> np.ndarray:
        return self._segment_references > 0

    @property
    def _free_points_in_total(self) -> int:
        return self.total_capacity - np.sum(self._segment_capacity[self._segment_reserved])

    @property
    def _free_points_at_end(self) -> int:
        reserved_index = np.flatnonzero(self._segment_reserved)
        if len(reserved_index):
            return self.total_capacity - np.sum(self._segment_capacity[:reserved_index[-1]])
        else:
            return self.total_capacity

    @with_select
    def read_waveforms(self) -> List[np.ndarray]:
        device = self.device._get_readable_device(simulator=True)

        old_segment = device.send_query(":TRAC:SEL?")
        waveforms = []
        uploaded_waveform_indices = np.flatnonzero(
            self._segment_references) + 1

        for segment in uploaded_waveform_indices:
            device.send_cmd(":TRAC:SEL {}".format(segment), paranoia_level=self.internal_paranoia_level)
            waveforms.append(device.read_segment_data())
        device.send_cmd(":TRAC:SEL {}".format(old_segment), paranoia_level=self.internal_paranoia_level)
        return waveforms

    @with_select
    def read_sequence_tables(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        device = self.device._get_readable_device(simulator=True)

        old_sequence = device.send_query(":SEQ:SEL?")
        sequences = []
        uploaded_sequence_indices = np.arange(len(self._sequencer_tables)) + 1
        for sequence in uploaded_sequence_indices:
            device.send_cmd(":SEQ:SEL {}".format(sequence), paranoia_level=self.internal_paranoia_level)
            table = device.read_sequencer_table()
            sequences.append((table['repeats'], table['segment_no'], table['jump_flag']))
        device.send_cmd(":SEQ:SEL {}".format(old_sequence), paranoia_level=self.internal_paranoia_level)
        return sequences

    @with_select
    def read_advanced_sequencer_table(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        table = self.device._get_readable_device(simulator=True).read_advanced_sequencer_table()
        return table['repeats'], table['segment_no'], table['jump_flag']

    def read_complete_program(self) -> PlottableProgram:
        return PlottableProgram.from_read_data(self.read_waveforms(),
                                               self.read_sequence_tables(),
                                               self.read_advanced_sequencer_table())

    def _find_place_for_segments_in_memory(self, segments: Sequence, segment_lengths: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # TODO: comment was not finished
        """
        1. Find known segments
        2. Find empty spaces with fitting length
        3. Find empty spaces with bigger length
        4. Amend remaining segments

        Args:
            segments (Sequence):
            segment_length (Sequence):

        Returns:

        """
        segment_hashes = np.fromiter((hash(segment) for segment in segments), count=len(segments), dtype=np.int64)

        waveform_to_segment = find_positions(self._segment_hashes, segment_hashes)

        # separate into known and unknown
        unknown = (waveform_to_segment == -1)
        known = ~unknown

        known_pos_in_memory = waveform_to_segment[known]

        assert len(known_pos_in_memory) == 0 or np.all(
            self._segment_hashes[known_pos_in_memory] == segment_hashes[known])

        new_reference_counter = self._segment_references.copy()
        new_reference_counter[known_pos_in_memory] += 1

        to_upload_size = np.sum(segment_lengths[unknown] + 16)
        free_points_in_total = self.total_capacity - np.sum(self._segment_capacity[self._segment_references > 0])
        if free_points_in_total < to_upload_size:
            raise MemoryError("Not enough free memory",
                              free_points_in_total,
                              to_upload_size,
                              self._free_points_in_total)

        to_amend = cast(np.ndarray, unknown)
        to_insert = np.full(len(segments), fill_value=-1, dtype=np.int64)

        reserved_indices = np.flatnonzero(new_reference_counter > 0)
        first_free = reserved_indices[-1] + 1 if len(reserved_indices) else 0

        free_segments = new_reference_counter[:first_free] == 0
        free_segment_count = np.sum(free_segments)

        # look for a free segment place with the same length
        for segment_idx in np.flatnonzero(to_amend):
            if free_segment_count == 0:
                break

            pos_of_same_length = np.logical_and(free_segments,
                                                segment_lengths[segment_idx] == self._segment_capacity[:first_free])
            idx_same_length = np.argmax(pos_of_same_length)
            if pos_of_same_length[idx_same_length]:
                free_segments[idx_same_length] = False
                free_segment_count -= 1

                to_amend[segment_idx] = False
                to_insert[segment_idx] = idx_same_length

        # try to find places that are larger than the segments to fit in starting with the large segments and large
        # free spaces
        segment_indices = np.flatnonzero(to_amend)[np.argsort(segment_lengths[to_amend])[::-1]]
        capacities = self._segment_capacity[:first_free]
        for segment_idx in segment_indices:
            free_capacities = capacities[free_segments]
            free_segments_indices = np.flatnonzero(free_segments)[np.argsort(free_capacities)[::-1]]

            if len(free_segments_indices) == 0:
                break

            fitting_segment = np.argmax((free_capacities >= segment_lengths[segment_idx])[::-1])
            fitting_segment = free_segments_indices[fitting_segment]
            if self._segment_capacity[fitting_segment] >= segment_lengths[segment_idx]:
                free_segments[fitting_segment] = False
                to_amend[segment_idx] = False
                to_insert[segment_idx] = fitting_segment

        free_points_at_end = self.total_capacity - np.sum(self._segment_capacity[:first_free])
        if np.sum(segment_lengths[to_amend] + 16) > free_points_at_end:
            raise MemoryError("Fragmentation does not allow upload.",
                              np.sum(segment_lengths[to_amend] + 16),
                              free_points_at_end,
                              self._free_points_at_end)

        return waveform_to_segment, to_amend, to_insert

    @with_select
    @with_configuration_guard
    def _upload_segment(self, segment_index: int, segment: TaborSegment) -> None:
        if self._segment_references[segment_index] > 0:
            raise ValueError("Reference count not zero")
        if segment.num_points > self._segment_capacity[segment_index]:
            raise ValueError("Cannot upload segment here.")

        segment_no = segment_index + 1

        self.device[TaborSCPI].send_cmd(":TRAC:DEF {}, {}".format(segment_no, segment.num_points),
                                        paranoia_level=self.internal_paranoia_level)
        self._segment_lengths[segment_index] = segment.num_points

        self.device[TaborSCPI].send_cmd(":TRAC:SEL {}".format(segment_no),
                                        paranoia_level=self.internal_paranoia_level)

        self.device[TaborSCPI].send_cmd(":TRAC:MODE COMB",
                                        paranoia_level=self.internal_paranoia_level)
        wf_data = segment.get_as_binary()

        self.device._send_binary_data(bin_dat=wf_data)
        self._segment_references[segment_index] = 1
        self._segment_hashes[segment_index] = hash(segment)

    @with_select
    @with_configuration_guard
    def _amend_segments(self, segments: List[TaborSegment]) -> np.ndarray:
        new_lengths = np.asarray([s.num_points for s in segments], dtype=np.uint32)

        wf_data = make_combined_wave(segments)
        trac_len = len(wf_data) // 2

        segment_index = len(self._segment_capacity)
        first_segment_number = segment_index + 1

        self.device[TaborSCPI].send_cmd(":TRAC:DEF {},{}".format(first_segment_number, trac_len),
                                        paranoia_level=self.internal_paranoia_level)
        self.device[TaborSCPI].send_cmd(":TRAC:SEL {}".format(first_segment_number),
                                        paranoia_level=self.internal_paranoia_level)
        self.device[TaborSCPI].send_cmd(":TRAC:MODE COMB",
                                        paranoia_level=self.internal_paranoia_level)
        self.device._send_binary_data(bin_dat=wf_data)

        old_to_update = np.count_nonzero(self._segment_capacity != self._segment_lengths)
        segment_capacity = np.concatenate((self._segment_capacity, new_lengths))
        segment_lengths = np.concatenate((self._segment_lengths, new_lengths))
        segment_references = np.concatenate((self._segment_references, np.ones(len(segments), dtype=int)))
        segment_hashes = np.concatenate((self._segment_hashes, [hash(s) for s in segments]))
        if len(segments) < old_to_update:
            for i, segment in enumerate(segments):
                current_segment_number = first_segment_number + i
                self.device[TaborSCPI].send_cmd(":TRAC:DEF {},{}".format(current_segment_number, segment.num_points),
                                                paranoia_level=self.internal_paranoia_level)
        else:
            # flush the capacity
            self.device._download_segment_lengths(segment_capacity)

            # update non fitting lengths
            for i in np.flatnonzero(segment_capacity != segment_lengths):
                self.device[SCPI].send_cmd(":TRAC:DEF {},{}".format(i + 1, segment_lengths[i]))

        self._segment_capacity = segment_capacity
        self._segment_lengths = segment_lengths
        self._segment_hashes = segment_hashes
        self._segment_references = segment_references

        return segment_index + np.arange(len(segments), dtype=np.int64)

    @with_select
    @with_configuration_guard
    def cleanup(self) -> None:
        """Discard all segments after the last which is still referenced"""
        reserved_indices = np.flatnonzero(self._segment_references > 0)
        old_end = len(self._segment_lengths)
        new_end = reserved_indices[-1] + 1 if len(reserved_indices) else 0
        self._segment_lengths = self._segment_lengths[:new_end]
        self._segment_capacity = self._segment_capacity[:new_end]
        self._segment_hashes = self._segment_hashes[:new_end]
        self._segment_references = self._segment_references[:new_end]

        try:
            #  send max 10 commands at once
            chunk_size = 10
            for chunk_start in range(new_end, old_end, chunk_size):
                self.device[SCPI].send_cmd("; ".join("TRAC:DEL {}".format(i + 1)
                                               for i in range(chunk_start, min(chunk_start + chunk_size, old_end))))
        except Exception as e:
            raise TaborUndefinedState("Error during cleanup. Device is in undefined state.", device=self) from e

    @with_configuration_guard
    def _execute_multiple_commands_with_config_guard(self, commands: List[str]) -> None:
        """ Joins the given commands into one and executes it with configuration guard.

        Args:
            commands: Commands that should be executed.
        """
        cmd_str = ";".join(commands)
        self.device[TaborSCPI].send_cmd(cmd_str, paranoia_level=self.internal_paranoia_level)

    def _enter_config_mode(self) -> None:
        """
        Enter the configuration mode if not already in. All outputs are set to the DC offset of the device and the
        sequencing is disabled. The manual states this speeds up sequence validation when uploading multiple sequences.
        When entering and leaving the configuration mode the AWG outputs a small (~60 mV in 4 V mode) blip.
        """
        if self._is_in_config_mode is False:

            # 1. Selct channel pair
            # 2. Select DC as function shape
            # 3. Select build-in waveform mode

            if self.device._is_coupled():
                out_cmd = ":OUTP:ALL OFF"
            else:
                out_cmd = ""
                for channel in self.channels:
                    out_cmd = out_cmd + ":INST:SEL {ch_id}; :OUTP OFF;".format(ch_id=channel.idn)

            marker_0_cmd = ":SOUR:MARK:SEL 1;:SOUR:MARK:SOUR USER;:SOUR:MARK:STAT OFF"
            marker_1_cmd = ":SOUR:MARK:SEL 2;:SOUR:MARK:SOUR USER;:SOUR:MARK:STAT OFF"

            wf_mode_cmd = ":SOUR:FUNC:MODE FIX"

            cmd = ";".join([marker_0_cmd, marker_1_cmd, wf_mode_cmd])
            cmd = out_cmd + cmd
            self.device[TaborSCPI].send_cmd(cmd, paranoia_level=self.CONFIG_MODE_PARANOIA_LEVEL)
            self._is_in_config_mode = True

    @with_select
    def _exit_config_mode(self) -> None:
        """Leave the configuration mode. Enter advanced sequence mode and turn on all outputs"""

        if self.device._is_coupled():
            # Coupled -> switch all channels at once
            other_channel_tuple: TaborChannelTuple
            if self.channels == self.device.channel_tuples[0].channels:
                other_channel_tuple = self.device.channel_tuples[1]
            else:
                other_channel_tuple = self.device.channel_tuples[0]

            if not other_channel_tuple._is_in_config_mode:
                self.device[SCPI].send_cmd(":SOUR:FUNC:MODE ASEQ")
                self.device[SCPI].send_cmd(":SEQ:SEL 1")
                self.device[SCPI].send_cmd(":OUTP:ALL ON")

        else:
            self.device[SCPI].send_cmd(":SOUR:FUNC:MODE ASEQ")
            self.device[SCPI].send_cmd(":SEQ:SEL 1")

            for channel in self.channels:
                channel[ActivatableChannels].enable()

        for marker_ch in self.marker_channels:
            marker_ch[ActivatableChannels].enable()

        self._is_in_config_mode = False


########################################################################################################################
# Marker Channel
########################################################################################################################
# Features

class TaborActivatableMarkerChannels(ActivatableChannels):
    def __init__(self, marker_channel: "TaborMarkerChannel"):
        super().__init__()
        self._parent = weakref.ref(marker_channel)

    @property
    def enabled(self) -> bool:
        """
        Returns the the state a marker channel has at the moment. A channel is either activated or deactivated
        True stands for activated and false for deactivated
        """
        return self._parent().device[SCPI].send_query(":MARK:STAT ?") == "ON"

    @with_select
    def enable(self):
        """Enables the output of a certain marker channel"""
        command_string = "SOUR:MARK:SOUR USER; :SOUR:MARK:STAT ON"
        command_string = command_string.format(
            channel=self._parent().channel_tuple.channels[0].idn,
            marker=self._parent().channel_tuple.marker_channels.index(self._parent()) + 1)
        self._parent().device[SCPI].send_cmd(command_string)

    @with_select
    def disable(self):
        """Disable the output of a certain marker channel"""
        command_string = ":SOUR:MARK:SOUR USER; :SOUR:MARK:STAT OFF"
        command_string = command_string.format(
            channel=self._parent().channel_tuple.channels[0].idn,
            marker=self._parent().channel_tuple.marker_channels.index(self._parent()) + 1)
        self._parent().device[SCPI].send_cmd(command_string)

    def _select(self) -> None:
        self._parent()._select()


# Implementation
class TaborMarkerChannel(AWGMarkerChannel):
    def __init__(self, idn: int, device: TaborDevice):
        super().__init__(idn)
        self._device = weakref.ref(device)

        # adding Features
        self.add_feature(TaborActivatableMarkerChannels(self))

    @property
    def device(self) -> TaborDevice:
        """Returns the device that this marker channel belongs to"""
        return self._device()

    @property
    def channel_tuple(self) -> TaborChannelTuple:
        """Returns the channel tuple that this marker channel belongs to"""
        return self._channel_tuple()

    def _set_channel_tuple(self, channel_tuple: TaborChannelTuple) -> None:
        """
        The channel tuple 'channel_tuple' is assigned to this marker channel

        Args:
            channel_tuple (TaborChannelTuple): the channel tuple that this marker channel belongs to
        """
        self._channel_tuple = weakref.ref(channel_tuple)

    def _select(self) -> None:
        """
        This marker channel is selected and is now the active channel marker of the device
        """
        self.device.channels[int((self.idn - 1) / 2)]._select()
        self.device[SCPI].send_cmd(":SOUR:MARK:SEL {marker}".format(marker=(((self.idn - 1) % 2) + 1)))


class TaborUndefinedState(TaborException):
    """
    If this exception is raised the attached tabor device is in an undefined state.
    It is highly recommended to call reset it.f
    """

    def __init__(self, *args, device: Union[TaborDevice, TaborChannelTuple]):
        super().__init__(*args)
        self.device = device

    def reset_device(self):
        if isinstance(self.device, TaborDevice):
            self.device[TaborDeviceControl].reset()
        elif isinstance(self.device, TaborChannelTuple):
            self.device.cleanup()
            self.device[TaborProgramManagement].clear()
