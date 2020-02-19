import functools
from typing import Optional, Set, Tuple, Callable, Dict, Union, Any, Iterable, List, NamedTuple
from collections import OrderedDict
import numpy as np
from qupulse import ChannelID
from qupulse._program._loop import Loop

from qupulse.hardware.awgs.features import ChannelSynchronization, AmplitudeOffsetHandling, OffsetAmplitude, \
    ProgramManagement
from qupulse.hardware.util import voltage_to_uint16, make_combined_wave
from qupulse.utils.types import Collection
from qupulse.hardware.awgs.base import AWGChannelTuple, AWGChannel, AWGDevice, AWGMarkerChannel
from typing import Sequence

# Provided by Tabor electronics for python 2.7
# a python 3 version is in a private repository on https://git.rwth-aachen.de/qutech
# Beware of the string encoding change!
import teawg
import numpy as np

from qupulse.hardware.awgs.features.amplitude_offset_feature import ChannelAmplitudeOffsetFeature
from qupulse.hardware.awgs.features.device_mirror_feature import DeviceMirrorFeature
from qupulse.utils.types import ChannelID
from qupulse._program._loop import Loop, make_compatible
from qupulse.hardware.util import voltage_to_uint16, make_combined_wave, find_positions, get_sample_times
from qupulse.hardware.awgs.old_base import AWG, AWGAmplitudeOffsetHandling


from qupulse.hardware.awgs.base import AWGDevice, AWGChannel, AWGChannelTuple


# TODO: ???
assert (sys.byteorder == 'little')


# TODO: ???
# __all__ = ['TaborAWGRepresentation', 'TaborChannelPair']

########################################################################################################################

class TaborSegment:
    pass #  TODO: to implement

class TaborProgram:
    pass #  TODO: to implement


# TODO: How does this work?
def with_configuration_guard(function_object: Callable[['TaborChannelTuple', Any], Any]) -> Callable[
    ['TaborChannelTuple'], Any]:
    """This decorator assures that the AWG is in configuration mode while the decorated method runs."""

    @functools.wraps(function_object)
    def guarding_method(channel_pair: 'TaborChannelTuple', *args, **kwargs) -> Any:
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


# TODO: How does this work?
def with_select(function_object: Callable[['TaborChannelTuple', Any], Any]) -> Callable[['TaborChannelTuple'], Any]:
    """Asserts the channel pair is selcted when the wrapped function is called"""

    @functools.wraps(function_object)
    def selector(channel_pair: 'TaborChannelTuple', *args, **kwargs) -> Any:
        channel_pair.select()
        return function_object(channel_pair, *args, **kwargs)

    return selector


TaborProgramMemory = NamedTuple('TaborProgramMemory', [('waveform_to_segment', np.ndarray),
                                                       ('program', TaborProgram)])

########################################################################################################################
# Device
########################################################################################################################
# Features
# TODO: implement Synchronization Feature for Tabor Devices
"""
class TaborChannelSynchronization(ChannelSynchronization):
    def __init__(self, device: "TaborDevice"):
        super().__init__()
        self._parent = device

    def synchronize_channels(self, group_size: int) -> None:
        pass  # TODO: to implement
"""


# Implementation
class TaborDevice(AWGDevice):
    def __init__(self, device_name: str, instr_addr=None, paranoia_level=1, external_trigger=False, reset=False,
                 mirror_addresses=()):
        """
        :param device_name:       Name of the device
        :param instr_addr:        Instrument address that is forwarded to teawag
        :param paranoia_level:    Paranoia level that is forwarded to teawg
        :param external_trigger:  Not supported yet
        :param reset:
        :param mirror_addresses:  addresses of multiple device which can be controlled at once
        """

        super().__init__(device_name)
        self._instr = teawg.TEWXAwg(instr_addr, paranoia_level)
        self._mirrors = tuple(teawg.TEWXAwg(address, paranoia_level) for address in mirror_addresses)
        self._coupled = None
        self._clock_marker = [0, 0, 0, 0]  # TODO: What are clock markers used for?


        # Channel
        self._channels = [TaborChannel(i + 1, self) for i in range(4)]

        # ChannelTuple TODO: ChannelMarker fehlen / bzw. Liste leer
        self._channel_tuples = []
        self._channel_tuples.append(TaborChannelTuple(1, self, self.channels[0:1], []))
        self._channel_tuples.append(TaborChannelTuple(2, self, self.channels[2:3], []))

        # ChannelMarker
        self._channel_marker = []

        if external_trigger:
            raise NotImplementedError()  # pragma: no cover

        if reset:
            self.send_cmd(':RES')

        self.initialize()  # TODO: ggf. ueberarbeiten

    # Trotzdem noch eine Ampitude Methode?

    # Trotzdem noch eine Offset Methode?

    def cleanup(self) -> None:
        pass  # TODO: to implement

    # TODO: Kann Collection auch noch spezialiseirt werden?
    @property
    def channels(self) -> Collection["TaborChannel"]:
        return self._channels

    @property
    def marker_channels(self) -> Collection["AWGMarkerChannel"]:
        return self._channel_marker

    @property
    def channel_tuples(self) -> Collection["AWGChannelTuple"]:
        return self._channel_tuples

    @property
    def main_instrument(self) -> teawg.TEWXAwg:
        return self._instr

    @property
    def mirrored_instruments(self) -> Sequence[teawg.TEWXAwg]:
        return self._mirrors

    @property
    def paranoia_level(self) -> int:
        return self._instr.paranoia_level

    @paranoia_level.setter
    def paranoia_level(self, val):
        for instr in self.all_devices:
            instr.paranoia_level = val

    @property
    def dev_properties(self) -> dict:
        return self._instr.dev_properties

    @property
    def all_devices(self) -> Sequence[teawg.TEWXAwg]:
        return (self._instr,) + self._mirrors

    def send_cmd(self, cmd_str, paranoia_level=None):
        print(self.all_devices)
        for instr in self.all_devices:
            instr.send_cmd(cmd_str=cmd_str, paranoia_level=paranoia_level)

    def send_query(self, query_str, query_mirrors=False) -> Any:
        if query_mirrors:
            return tuple(instr.send_query(query_str) for instr in self.all_devices)
        else:
            return self._instr.send_query(query_str)

    def send_binary_data(self, pref, bin_dat, paranoia_level=None):
        for instr in self.all_devices:
            instr.send_binary_data(pref, bin_dat=bin_dat, paranoia_level=paranoia_level)

    def download_segment_lengths(self, seg_len_list, pref=':SEGM:DATA', paranoia_level=None):
        for instr in self.all_devices:
            instr.download_segment_lengths(seg_len_list, pref=pref, paranoia_level=paranoia_level)

    def download_sequencer_table(self, seq_table, pref=':SEQ:DATA', paranoia_level=None):
        for instr in self.all_devices:
            instr.download_sequencer_table(seq_table, pref=pref, paranoia_level=paranoia_level)

    def download_adv_seq_table(self, seq_table, pref=':ASEQ:DATA', paranoia_level=None):
        for instr in self.all_devices:
            instr.download_adv_seq_table(seq_table, pref=pref, paranoia_level=paranoia_level)

    make_combined_wave = staticmethod(teawg.TEWXAwg.make_combined_wave)

    def _send_cmd(self, cmd_str, paranoia_level=None) -> Any:
        """Overwrite send_cmd for paranoia_level > 3"""
        if paranoia_level is None:
            paranoia_level = self.paranoia_level

        if paranoia_level < 3:
            # TODO: unsolved Reference
            super().send_cmd(cmd_str=cmd_str, paranoia_level=paranoia_level)  # pragma: no cover
        else:
            cmd_str = cmd_str.rstrip()

            if len(cmd_str) > 0:
                ask_str = cmd_str + '; *OPC?; :SYST:ERR?'
            else:
                ask_str = '*OPC?; :SYST:ERR?'

            *answers, opc, error_code_msg = self._visa_inst.ask(ask_str).split(';')

            error_code, error_msg = error_code_msg.split(',')
            error_code = int(error_code)
            if error_code != 0:
                _ = self._visa_inst.ask('*CLS; *OPC?')

                if error_code == -450:
                    # query queue overflow
                    self.send_cmd(cmd_str)
                else:
                    raise RuntimeError('Cannot execute command: {}\n{}: {}'.format(cmd_str, error_code, error_msg))

            assert len(answers) == 0

    def get_status_table(self) -> Dict[str, Union[str, float, int]]:
        """Send a lot of queries to the AWG about its settings. A good way to visualize is using pandas.DataFrame

                Returns:
                    An ordered dictionary with the results
                """
        name_query_type_list = [('channel', ':INST:SEL?', int),
                                ('coupling', ':OUTP:COUP?', str),
                                ('volt_dc', ':SOUR:VOLT:LEV:AMPL:DC?', float),
                                ('volt_hv', ':VOLT:HV?', float),
                                ('offset', ':VOLT:OFFS?', float),
                                ('outp', ':OUTP?', str),
                                ('mode', ':SOUR:FUNC:MODE?', str),
                                ('shape', ':SOUR:FUNC:SHAPE?', str),
                                ('dc_offset', ':SOUR:DC?', float),
                                ('freq_rast', ':FREQ:RAST?', float),

                                ('gated', ':INIT:GATE?', str),
                                ('continuous', ':INIT:CONT?', str),
                                ('continuous_enable', ':INIT:CONT:ENAB?', str),
                                ('continuous_source', ':INIT:CONT:ENAB:SOUR?', str),
                                ('marker_source', ':SOUR:MARK:SOUR?', str),
                                ('seq_jump_event', ':SOUR:SEQ:JUMP:EVEN?', str),
                                ('seq_adv_mode', ':SOUR:SEQ:ADV?', str),
                                ('aseq_adv_mode', ':SOUR:ASEQ:ADV?', str),

                                ('marker', ':SOUR:MARK:SEL?', int),
                                ('marker_high', ':MARK:VOLT:HIGH?', str),
                                ('marker_low', ':MARK:VOLT:LOW?', str),
                                ('marker_width', ':MARK:WIDT?', int),
                                ('marker_state', ':MARK:STAT?', str)]

        data = OrderedDict((name, []) for name, *_ in name_query_type_list)
        for ch in (1, 2, 3, 4):
            # TODO: select Channel und Marker fehlen im Device
            self._select_channel(ch)
            self.select_marker((ch - 1) % 2 + 1)
            for name, query, dtype in name_query_type_list:
                data[name].append(dtype(self.send_query(query)))
        return data

    @property
    def is_open(self) -> bool:
        return self._instr.visa_inst is not None  # pragma: no cover

    # TODO: soll man ein Channel Objekt oder eine ChannelNummer mitgeben? -> intern, das was am besten fuer die Umsetzung ist
    def _select_channel(self, channel_nr: int) -> None:
        if channel_nr not in range(1, len(self.channels)):
            raise TaborException('Invalid channel: {}'.format(channel_nr))

        self.send_cmd(':INST:SEL {channel}'.format(channel=channel_nr))

    def _select_marker(self, marker_nr: int) -> None:
        # TODO: right name for the parameter?
        """Select marker a marker of the currently active channel pair."""
        if marker_nr not in range(1, len(self.channel_tuples[1].marker_channels)):
            raise TaborException('Invalid marker: {}'.format(marker_nr))

        self.send_cmd(':SOUR:MARK:SEL {marker}'.format(marker=marker_nr))

    # wird die Methode noch gebraucht?
    def _sample_rate(self, channel_nr: int) -> int:
        if channel_nr not in range(1, len(self.channels)):
            raise TaborException('Invalid channel: {}'.format(channel_nr))

        return int(self.channels[channel_nr].channel_tuple.sample_rate)

    # def setter_sample_rate implementieren?

    def enable(self) -> None:
        self.send_cmd(':ENAB')

    def abort(self) -> None:
        self.send_cmd(':ABOR')

    def initialize(self) -> None:
        # 1. Select channel
        # 2. Turn off gated mode
        # 3. continous mode
        # 4. Armed mode (onlz generate waveforms after enab command)
        # 5. Expect enable signal from (USB / LAN / GPIB)
        # 6. Use arbitrary waveforms as marker source
        # 7. Expect jump command for sequencing from (USB / LAN / GPIB)
        setup_command = (
            ":INIT:GATE OFF; :INIT:CONT ON; "
            ":INIT:CONT:ENAB SELF; :INIT:CONT:ENAB:SOUR BUS; "
            ":SOUR:MARK:SOUR USER; :SOUR:SEQ:JUMP:EVEN BUS ")
        self.send_cmd(':INST:SEL 1')
        self.send_cmd(setup_command)
        self.send_cmd(':INST:SEL 3')
        self.send_cmd(setup_command)

    def reset(self) -> None:
        self.send_cmd(':RES')
        self._coupled = None
        self.initialize()
        for channel_tuple in self.channel_tuples:
            channel_tuple.clear()

    def trigger(self) -> None:
        self.send_cmd(':TRIG')

    def get_readable_device(self, simulator=True) -> teawg.TEWXAwg:
        for device in self.all_devices:
            if device.fw_ver >= 3.0:
                if simulator:
                    if device.is_simulator:
                        return device
                else:
                    return device
        raise TaborException('No device capable of device data read')


########################################################################################################################
# Channel
########################################################################################################################
# Features
class TaborOffsetAmplitude(OffsetAmplitude):
    def __init__(self, channel: "TaborChannel"):
        super().__init__()
        self._parent = channel

    @property
    def offset(self) -> float:
        return float(
            self._parent.device.send_query(':INST:SEL {channel}; :VOLT:OFFS?'.format(channel=self._parent.idn)))

    @offset.setter
    def offset(self, offset: float) -> None:
        pass  # TODO: to implement

    @property
    def amplitude(self) -> float:
        coupling = self._parent.device.send_query(':INST:SEL {channel}; :OUTP:COUP?'.format(channel=self._parent.idn))
        if coupling == 'DC':
            return float(self._parent.device.send_query(':VOLT?'))
        elif coupling == 'HV':
            return float(self._parent.device.send_query(':VOLT:HV?'))
        else:
            raise TaborException('Unknown coupling: {}'.format(coupling))

    @amplitude.setter
    def amplitude(self, amplitude: float) -> None:
        pass  # TODO: to implement

    @property
    def amplitude_offset_handling(self) -> str:
        pass  # TODO: to implement

    @amplitude_offset_handling.setter
    def amplitude_offset_handling(self, amp_offs_handling: str) -> None:
        pass  # TODO: to implement


# Implementation
class TaborChannel(AWGChannel):
    def __init__(self, idn: int, device: TaborDevice):
        super().__init__(idn)

        self._device = device

        self.add_feature(TaborOffsetAmplitude(self))

    @property
    def device(self) -> TaborDevice:
        return self._device

    @property
    def channel_tuple(self) -> Optional[AWGChannelTuple]:
        pass

    def _set_channel_tuple(self, channel_tuple) -> None:
        pass


########################################################################################################################
# ChannelTuple
########################################################################################################################
# Features
class TaborProgramManagement(ProgramManagement):
    def __init__(self, channel_tuple: "TaborChannelTuple", ):
        super().__init__()
        self._programs = {}
        self._armed_program = None
        self._parent = channel_tuple

    def upload(self, name: str, program: Loop, channels: Tuple[Optional[ChannelID], ...],
               markers: Tuple[Optional[ChannelID], ...], voltage_transformation: Tuple[Optional[Callable], ...],
               force: bool = False) -> None:
        pass  # TODO: to implement

    def remove(self, name: str) -> None:
        """Remove a program from the AWG.

        Also discards all waveforms referenced only by the program identified by name.

        Args:
            name (str): The name of the program to remove.
        """
        self._parent.free_program(name)
        self._parent.cleanup()

    def clear(self) -> None:
        pass  # TODO: to implement

    def arm(self, name: Optional[str]) -> None:
        self._parent._arm()

    @property
    def programs(self) -> Set[str]:
        pass  # TODO: to implement


class TaborChannelTuple(AWGChannelTuple):
    def __init__(self, idn: int, device: TaborDevice, channels: Iterable["TaborChannel"],
                 marker_channels: Iterable["TaborMarkerChannel"]):
        # TODO: hat das weglassen des alten String identifier Auswirkungen?
        # TODO: zugeordneter MarkerChannel

        super().__init__(idn)
        self._device = device  # TODO: weakref.ref(device) can't be used like in  the old driver

        self._configuration_guard_count = 0
        self._is_in_config_mode = False

        # TODO: Ueberpreufung macht keinen Sinn
        #if channels not in self._device.channel_tuples:
        #    raise ValueError('Invalid channel pair: {}'.format(channels))
        self._channels = tuple(channels)

        self._marker_channels = tuple(marker_channels)

        self.add_feature(TaborProgramManagement(self))

        #TODO: Kommentar beenden
        """
        self._idle_segment = TaborSegment(voltage_to_uint16(voltage=np.zeros(192),
                                                            output_amplitude=0.5,
                                                            output_offset=0., resolution=14),
                                          voltage_to_uint16(voltage=np.zeros(192),
                                                            output_amplitude=0.5,
                                                            output_offset=0., resolution=14), None, None)
        """
        self._idle_sequence_table = [(1, 1, 0), (1, 1, 0), (1, 1, 0)]

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


    @property
    def device(self) -> TaborDevice:
        return self._device

    @property
    def channels(self) -> Collection["AWGChannel"]:
        return self._channels


    @property
    def marker_channels(self) -> Collection["AWGMarkerChannel"]:
        return self._marker_channels

    @property
    def sample_rate(self) -> float:
        pass  # TODO: to implement

    def select(self) -> None:
        pass  # TODO: to implement

    @property
    def total_capacity(self) -> int:
        return int(self.device.dev_properties['max_arb_mem']) // 2

    def free_program(self, name: str) -> TaborProgramMemory:
        pass  # TODO: to implement

    def _restore_program(self) -> None:
        pass  # TODO: to implement

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
        device = self.device.get_readable_device(simulator=True)

        old_segment = device.send_query(':TRAC:SEL?')
        waveforms = []
        uploaded_waveform_indices = np.flatnonzero(self._segment_references) + 1
        for segment in uploaded_waveform_indices:
            device.send_cmd(':TRAC:SEL {}'.format(segment), paranoia_level=self.internal_paranoia_level)
            waveforms.append(device.read_act_seg_dat())
        device.send_cmd(':TRAC:SEL {}'.format(old_segment), paranoia_level=self.internal_paranoia_level)
        return waveforms

    @with_select
    def read_sequence_tables(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        device = self.device.get_readable_device(simulator=True)

        old_sequence = device.send_query(':SEQ:SEL?')
        sequences = []
        uploaded_sequence_indices = np.arange(len(self._sequencer_tables)) + 1
        for sequence in uploaded_sequence_indices:
            device.send_cmd(':SEQ:SEL {}'.format(sequence), paranoia_level=self.internal_paranoia_level)
            sequences.append(device.read_sequencer_table())
        device.send_cmd(':SEQ:SEL {}'.format(old_sequence), paranoia_level=self.internal_paranoia_level)
        return sequences

    # upload im Feature

    def read_complete_program(self) -> PlottableProgram:
        return PlottableProgram.from_read_data(self.read_waveforms(),
                                               self.read_sequence_tables(),
                                               self.read_advanced_sequencer_table())

    # clear im Feature

    def _find_place_for_segments_in_memory(self, segments: Sequence, segment_lengths: Sequence) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        pass  # TODO: to implement

    @with_select
    @with_configuration_guard
    def _upload_segment(self, segment_index: int, segment: TaborSegment) -> None:
        #  TODO: Why is the proptery for device not used?
        if self._segment_references[segment_index] > 0:
            raise ValueError('Reference count not zero')
        if segment.num_points > self._segment_capacity[segment_index]:
            raise ValueError('Cannot upload segment here.')

        segment_no = segment_index + 1

        self.device.send_cmd(':TRAC:DEF {}, {}'.format(segment_no, segment.num_points),
                             paranoia_level=self.internal_paranoia_level)
        self._segment_lengths[segment_index] = segment.num_points

        self.device.send_cmd(':TRAC:SEL {}'.format(segment_no), paranoia_level=self.internal_paranoia_level)

        self.device.send_cmd(':TRAC:MODE COMB', paranoia_level=self.internal_paranoia_level)
        wf_data = segment.get_as_binary()

        self.device.send_binary_data(pref=':TRAC:DATA', bin_dat=wf_data)
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
        self.device.send_cmd(':TRAC:DEF {},{}'.format(first_segment_number, trac_len),
                             paranoia_level=self.internal_paranoia_level)
        self.device.send_cmd(':TRAC:SEL {}'.format(first_segment_number),
                             paranoia_level=self.internal_paranoia_level)
        self.device.send_cmd(':TRAC:MODE COMB',
                             paranoia_level=self.internal_paranoia_level)
        self.device.send_binary_data(pref=':TRAC:DATA', bin_dat=wf_data)

        old_to_update = np.count_nonzero(self._segment_capacity != self._segment_lengths)
        segment_capacity = np.concatenate((self._segment_capacity, new_lengths))
        segment_lengths = np.concatenate((self._segment_lengths, new_lengths))
        segment_references = np.concatenate((self._segment_references, np.ones(len(segments), dtype=int)))
        segment_hashes = np.concatenate((self._segment_hashes, [hash(s) for s in segments]))
        if len(segments) < old_to_update:
            for i, segment in enumerate(segments):
                current_segment_number = first_segment_number + i
                self.device.send_cmd(':TRAC:DEF {},{}'.format(current_segment_number, segment.num_points),
                                     paranoia_level=self.internal_paranoia_level)
        else:
            # flush the capacity
            self.device.download_segment_lengths(segment_capacity)

            # update non fitting lengths
            for i in np.flatnonzero(segment_capacity != segment_lengths):
                self.device.send_cmd(':TRAC:DEF {},{}'.format(i + 1, segment_lengths[i]))

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
                self.device.send_cmd('; '.join('TRAC:DEL {}'.format(i + 1)
                                               for i in range(chunk_start, min(chunk_start + chunk_size, old_end))))
        except Exception as e:
            raise TaborUndefinedState('Error during cleanup. Device is in undefined state.', device=self) from e

    # remove im Feature

    @with_configuration_guard
    def _execute_multiple_commands_with_config_guard(self, commands: List[str]) -> None:
        """ Joins the given commands into one and executes it with configuration guard.

        Args:
            commands: Commands that should be executed.
        """
        cmd_str = ";".join(commands)
        self.device.send_cmd(cmd_str, paranoia_level=self.internal_paranoia_level)

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

        waveform_to_segment_index, program = self._known_programs[program_name]

        modifications = program.update_volatile_parameters(parameters)

        self.logger.debug("parameter modifications: %r" % modifications)

        if not modifications:
            self.logger.info("There are no volatile parameters to update. Either there are no volatile parameters with "
                             "these names,\nthe respective repetition counts already have the given values or the "
                             "volatile parameters were dropped during upload.")
            return

        if program_name == self._current_program:
            commands = []

            for position, entry in modifications.items():
                if not entry.repetition_count > 0:
                    raise ValueError('Repetition must be > 0')

                if isinstance(position, int):
                    commands.append(":ASEQ:DEF {},{},{},{}".format(position + 1, entry.element_number + 1,
                                                                   entry.repetition_count, entry.jump_flag))
                else:
                    table_num, step_num = position
                    commands.append(":SEQ:SEL {}".format(table_num + 2))
                    commands.append(":SEQ:DEF {},{},{},{}".format(step_num,
                                                                  waveform_to_segment_index[entry.element_id] + 1,
                                                                  entry.repetition_count, entry.jump_flag))
            self._execute_multiple_commands_with_config_guard(commands)

        # Wait until AWG is finished
        _ = self.device.main_instrument._visa_inst.query('*OPC?')

    def set_marker_state(self, marker: int, active: bool) -> None:
        pass  # TODO: to implement

    def set_channel_state(self, channel, active) -> None:
        pass  # TODO: to implement

    @with_select
    def _arm(self, name: str) -> None:
        if self._current_program == name:
            self.device.send_cmd('SEQ:SEL 1', paranoia_level=self.internal_paranoia_level)
        else:
            self.change_armed_program(name)

    def set_program_advanced_sequence_table(self, name, new_advanced_sequence_table):
        self._known_programs[name][1]._advanced_sequencer_table = new_advanced_sequence_table

    def set_program_sequence_table(self, name, new_sequence_table):
        self._known_programs[name][1]._sequencer_tables = new_sequence_table

    @with_select
    @with_configuration_guard
    def change_armed_program(self, name: Optional[str]) -> None:
        pass  # TODO: to implement

    @with_select
    def run_current_program(self) -> None:
        if self._current_program:
            self.device.send_cmd(':TRIG', paranoia_level=self.internal_paranoia_level)
        else:
            raise RuntimeError('No program active')

    @property
    def programs(self) -> Set[str]:
        """The set of program names that can currently be executed on the hardware AWG."""
        return set(program for program in self._known_programs.keys())

    @property
    def sample_rate(self) -> float:
        pass  # TODO: to implement

    @property
    def num_channels(self) -> int:
        return len(self.channels)

    @property
    def num_markers(self) -> int:
        pass  # TODO: to implement

    def _enter_config_mode(self) -> None:
        """Enter the configuration mode if not already in. All outputs are set to the DC offset of the device and the
        sequencing is disabled. The manual states this speeds up sequence validation when uploading multiple sequences.
        When entering and leaving the configuration mode the AWG outputs a small (~60 mV in 4 V mode) blip."""
        if self._is_in_config_mode is False:

            # 1. Selct channel pair
            # 2. Select DC as function shape
            # 3. Select build-in waveform mode

            if self.device.is_coupled():
                out_cmd = ':OUTP:ALL OFF'
            else:
                self.device.send_cmd(':INST:SEL {}; :OUTP OFF; :INST:SEL {}; :OUTP OFF'.format(*self._channels))

            self.set_marker_state(0, False)
            self.set_marker_state(1, False)
            self.device.send_cmd(':SOUR:FUNC:MODE FIX')


            wf_mode_cmd = ':SOUR:FUNC:MODE FIX'

            cmd = ';'.join([out_cmd, marker_0_cmd, marker_1_cmd, wf_mode_cmd])
            self.device.send_cmd(cmd, paranoia_level=self.CONFIG_MODE_PARANOIA_LEVEL)
            self._is_in_config_mode = True

    @with_select
    def _exit_config_mode(self) -> None:
        pass  # TODO: to implement


########################################################################################################################
# Marker Channel
########################################################################################################################
class TaborMarkerChannel(AWGMarkerChannel):
    def __init__(self, idn: int, device: TaborDevice, channel_tuple: TaborChannelTuple):
        super().__init__(idn)
        self._device = device
        self._channel_tuple = channel_tuple

    @property
    def device(self) -> AWGDevice:
        pass

    @property
    def channel_tuple(self) -> Optional[AWGChannelTuple]:
        pass

    def _set_channel_tuple(self, channel_tuple) -> None:
        pass


########################################################################################################################
class TaborException(Exception):
    pass


class TaborUndefinedState(TaborException):
    """If this exception is raised the attached tabor device is in an undefined state.
    It is highly recommended to call reset it."""
    pass  # TODO: to implement
