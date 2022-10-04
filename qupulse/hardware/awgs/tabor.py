import fractions
import functools
import warnings
import weakref
import logging
import numbers
from typing import List, Tuple, Set, Callable, Optional, Any, Sequence, cast, Union, Dict, Mapping, NamedTuple
from collections import OrderedDict

import tabor_control.device
import numpy as np

from qupulse.utils.types import ChannelID
from qupulse._program._loop import Loop, make_compatible
from qupulse.hardware.util import voltage_to_uint16, find_positions, traced
from qupulse.hardware.awgs.base import AWG, AWGAmplitudeOffsetHandling
from qupulse._program.tabor import TaborSegment, TaborException, TaborProgram, PlottableProgram, TaborSequencing,\
    make_combined_wave


__all__ = ['TaborAWGRepresentation', 'TaborChannelPair']


@traced
class TaborAWGRepresentation:
    def __init__(self, instr_addr=None, paranoia_level=1, external_trigger=False, reset=False, mirror_addresses=()):
        """
        :param instr_addr:        Instrument address that is forwarded to tabor_control
        :param paranoia_level:    Paranoia level that is forwarded to tabor_control
        :param external_trigger:  Not supported yet
        :param reset:
        :param mirror_addresses:
        """
        visa_instr = tabor_control.open_session(instr_addr)
        paranoia_level = tabor_control.ParanoiaLevel(paranoia_level)

        self._instr = tabor_control.device.TEWXAwg(visa_instr, paranoia_level)
        self._mirrors = tuple(tabor_control.device.TEWXAwg(tabor_control.open_session(address), paranoia_level)
                              for address in mirror_addresses)
        self._coupled = None

        self._clock_marker = [0, 0, 0, 0]

        if external_trigger:
            raise NotImplementedError()  # pragma: no cover

        if reset:
            self.send_cmd(':RES')

        self.initialize()

        self._channel_pair_AB = TaborChannelPair(self, (1, 2), str(instr_addr) + '_AB')
        self._channel_pair_CD = TaborChannelPair(self, (3, 4), str(instr_addr) + '_CD')

    def is_coupled(self) -> bool:
        if self._coupled is None:
            return self.send_query(':INST:COUP:STAT?') == 'ON'
        else:
            return self._coupled

    @property
    def channel_pair_AB(self) -> 'TaborChannelPair':
        return self._channel_pair_AB

    @property
    def channel_pair_CD(self) -> 'TaborChannelPair':
        return self._channel_pair_CD

    @property
    def main_instrument(self) -> tabor_control.device.TEWXAwg:
        return self._instr

    @property
    def mirrored_instruments(self) -> Sequence[tabor_control.device.TEWXAwg]:
        return self._mirrors

    @property
    def paranoia_level(self) -> int:
        return self._instr.paranoia_level.value

    @paranoia_level.setter
    def paranoia_level(self, val):
        if isinstance(val, int):
            val = min(max(val, 0), 2)
        for instr in self.all_devices:
            instr.paranoia_level = val

    @property
    def dev_properties(self) -> dict:
        return self._instr.dev_properties.as_dict()

    @property
    def all_devices(self) -> Sequence[tabor_control.device.TEWXAwg]:
        return (self._instr, ) + self._mirrors

    def send_cmd(self, cmd_str, paranoia_level=None):
        for instr in self.all_devices:
            instr.send_cmd(cmd_str=cmd_str, paranoia_level=paranoia_level)

    def send_query(self, query_str, query_mirrors=False) -> Any:
        if query_mirrors:
            return tuple(instr.send_query(query_str) for instr in self.all_devices)
        else:
            return self._instr.send_query(query_str)

    def send_binary_data(self, pref, bin_dat, paranoia_level=None):
        assert pref == ':TRAC:DATA'
        for instr in self.all_devices:
            instr.write_segment_data(bin_dat, paranoia_level=paranoia_level)

    def download_segment_lengths(self, seg_len_list, pref=':SEGM:DATA', paranoia_level=None):
        assert pref == ':SEGM:DATA'
        for instr in self.all_devices:
            instr.write_segment_lengths(seg_len_list, paranoia_level=paranoia_level)

    def download_sequencer_table(self, seq_table, pref=':SEQ:DATA', paranoia_level=None):
        assert pref == ':SEQ:DATA'
        for instr in self.all_devices:
            instr.write_sequencer_table(seq_table, paranoia_level=paranoia_level)

    def download_adv_seq_table(self, seq_table, pref=':ASEQ:DATA', paranoia_level=None):
        assert pref == ':ASEQ:DATA'
        for instr in self.all_devices:
            instr.write_advanced_sequencer_table(seq_table, paranoia_level=paranoia_level)

    def _send_cmd(self, cmd_str, paranoia_level=None) -> Any:
        """Overwrite send_cmd for paranoia_level > 3"""
        if paranoia_level is None:
            paranoia_level = self.paranoia_level

        if paranoia_level < 3:
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
            self.select_channel(ch)
            self.select_marker((ch-1) % 2 + 1)
            for name, query, dtype in name_query_type_list:
                data[name].append(dtype(self.send_query(query)))
        return data

    @property
    def is_open(self) -> bool:
        return self._instr.visa_inst is not None  # pragma: no cover

    def select_channel(self, channel) -> None:
        if channel not in (1, 2, 3, 4):
            raise TaborException('Invalid channel: {}'.format(channel))

        self.send_cmd(':INST:SEL {channel}'.format(channel=channel))

    def select_marker(self, marker: int) -> None:
        """Select marker 1 or 2 of the currently active channel pair."""
        if marker not in (1, 2):
            raise TaborException('Invalid marker: {}'.format(marker))
        self.send_cmd(':SOUR:MARK:SEL {marker}'.format(marker=marker))

    def sample_rate(self, channel) -> int:
        if channel not in (1, 2, 3, 4):
            raise TaborException('Invalid channel: {}'.format(channel))
        return int(float(self.send_query(':INST:SEL {channel}; :FREQ:RAST?'.format(channel=channel))))

    def amplitude(self, channel) -> float:
        if channel not in (1, 2, 3, 4):
            raise TaborException('Invalid channel: {}'.format(channel))
        coupling = self.send_query(':INST:SEL {channel}; :OUTP:COUP?'.format(channel=channel))
        if coupling == 'DC':
            return float(self.send_query(':VOLT?'))
        elif coupling == 'HV':
            return float(self.send_query(':VOLT:HV?'))
        else:
            raise TaborException('Unknown coupling: {}'.format(coupling))

    def offset(self, channel: int) -> float:
        if channel not in (1, 2, 3, 4):
            raise TaborException('Invalid channel: {}'.format(channel))
        return float(self.send_query(':INST:SEL {channel}; :VOLT:OFFS?'.format(channel=channel)))

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
        self.channel_pair_AB.clear()
        self.channel_pair_CD.clear()

    def trigger(self) -> None:
        self.send_cmd(':TRIG')

    def get_readable_device(self, simulator=True) -> tabor_control.device.TEWXAwg:
        for device in self.all_devices:
            if device.supports_basic_reading():
                if simulator:
                    if device.is_simulator:
                        return device
                else:
                    return device
        raise TaborException('No device capable of device data read')


TaborProgramMemory = NamedTuple('TaborProgramMemory', [('waveform_to_segment', np.ndarray),
                                                       ('program', TaborProgram)])


def with_configuration_guard(function_object: Callable[['TaborChannelPair', Any], Any]) -> Callable[['TaborChannelPair'],
                                                                                                    Any]:
    """This decorator assures that the AWG is in configuration mode while the decorated method runs."""
    @functools.wraps(function_object)
    def guarding_method(channel_pair: 'TaborChannelPair', *args, **kwargs) -> Any:
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


def with_select(function_object: Callable[['TaborChannelPair', Any], Any]) -> Callable[['TaborChannelPair'], Any]:
    """Asserts the channel pair is selected when the wrapped function is called"""
    @functools.wraps(function_object)
    def selector(channel_pair: 'TaborChannelPair', *args, **kwargs) -> Any:
        channel_pair.select()
        return function_object(channel_pair, *args, **kwargs)

    return selector


@traced
class TaborChannelPair(AWG):
    CONFIG_MODE_PARANOIA_LEVEL = None

    def __init__(self, tabor_device: TaborAWGRepresentation, channels: Tuple[int, int], identifier: str):
        super().__init__(identifier)
        self._device = weakref.ref(tabor_device)

        self._configuration_guard_count = 0
        self._is_in_config_mode = False

        if channels not in ((1, 2), (3, 4)):
            raise ValueError('Invalid channel pair: {}'.format(channels))
        self._channels = channels

        self._idle_segment = TaborSegment.from_sampled(voltage_to_uint16(voltage=np.zeros(192),
                                                                         output_amplitude=0.5,
                                                                         output_offset=0., resolution=14),
                                                       voltage_to_uint16(voltage=np.zeros(192),
                                                                         output_amplitude=0.5,
                                                                         output_offset=0., resolution=14),
                                                       None, None)
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

        self.clear()

    @property
    def internal_paranoia_level(self) -> Optional[int]:
        return self._internal_paranoia_level

    @internal_paranoia_level.setter
    def internal_paranoia_level(self, paranoia_level: Optional[int]):
        """ Sets the paranoia level with which commands from within methods are called """
        assert paranoia_level in (None, 0, 1, 2)
        self._internal_paranoia_level = paranoia_level

    def select(self) -> None:
        self.device.send_cmd(':INST:SEL {}'.format(self._channels[0]),
                             paranoia_level=self.internal_paranoia_level)

    @property
    def total_capacity(self) -> int:
        return int(self.device.dev_properties['max_arb_mem']) // 2

    @property
    def logger(self):
        return logging.getLogger("qupulse.tabor")

    @property
    def device(self) -> TaborAWGRepresentation:
        return self._device()

    def free_program(self, name: str) -> TaborProgramMemory:
        if name is None:
            raise TaborException('Removing "None" program is forbidden.')
        program = self._known_programs.pop(name)
        self._segment_references[program.waveform_to_segment] -= 1
        if self._current_program == name:
            self.change_armed_program(None)
        return program

    def _restore_program(self, name: str, program: TaborProgram) -> None:
        if name in self._known_programs:
            raise ValueError('Program cannot be restored as it is already known.')
        self._segment_references[program.waveform_to_segment] += 1
        self._known_programs[name] = program

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

        try:
            waveforms = []
            uploaded_waveform_indices = np.flatnonzero(self._segment_references) + 1
            for segment in uploaded_waveform_indices:
                device.send_cmd(':TRAC:SEL {}'.format(segment), paranoia_level=self.internal_paranoia_level)
                waveforms.append(device.read_segment_data())
        finally:
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
            sequencer_table = device.read_sequencer_table()
            sequences.append((sequencer_table['repeats'], sequencer_table['segment_no'], sequencer_table['jump_flag']))
        device.send_cmd(':SEQ:SEL {}'.format(old_sequence), paranoia_level=self.internal_paranoia_level)
        return sequences

    @with_select
    def read_advanced_sequencer_table(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        adv_seq_table = self.device.get_readable_device(simulator=True).read_advanced_sequencer_table()
        return adv_seq_table['repeats'], adv_seq_table['segment_no'], adv_seq_table['jump_flag']

    def read_complete_program(self) -> PlottableProgram:
        return PlottableProgram.from_read_data(self.read_waveforms(),
                                               self.read_sequence_tables(),
                                               self.read_advanced_sequencer_table())

    @with_configuration_guard
    @with_select
    def upload(self, name: str,
               program: Loop,
               channels: Tuple[Optional[ChannelID], Optional[ChannelID]],
               markers: Tuple[Optional[ChannelID], Optional[ChannelID]],
               voltage_transformation: Tuple[Callable, Callable],
               force: bool = False) -> None:
        """Upload a program to the AWG.

        The policy is to prefer amending the unknown waveforms to overwriting old ones."""

        if len(channels) != self.num_channels:
            raise ValueError('Channel ID not specified')
        if len(markers) != self.num_markers:
            raise ValueError('Markers not specified')
        if len(voltage_transformation) != self.num_channels:
            raise ValueError('Wrong number of voltage transformations')

        # adjust program to fit criteria
        sample_rate = self.device.sample_rate(self._channels[0])
        make_compatible(program,
                        minimal_waveform_length=192,
                        waveform_quantum=16,
                        sample_rate=fractions.Fraction(sample_rate, 10**9))

        if name in self._known_programs:
            if force:
                self.free_program(name)
            else:
                raise ValueError('{} is already known on {}'.format(name, self.identifier))

        # They call the peak to peak range amplitude
        ranges = (self.device.amplitude(self._channels[0]),
                  self.device.amplitude(self._channels[1]))

        voltage_amplitudes = (ranges[0]/2, ranges[1]/2)

        if self._amplitude_offset_handling == AWGAmplitudeOffsetHandling.IGNORE_OFFSET:
            voltage_offsets = (0, 0)
        elif self._amplitude_offset_handling == AWGAmplitudeOffsetHandling.CONSIDER_OFFSET:
            voltage_offsets = (self.device.offset(self._channels[0]),
                               self.device.offset(self._channels[1]))
        else:
            raise ValueError('{} is invalid as AWGAmplitudeOffsetHandling'.format(self._amplitude_offset_handling))

        # parse to tabor program
        tabor_program = TaborProgram(program,
                                     channels=tuple(channels),
                                     markers=markers,
                                     device_properties=self.device.dev_properties,
                                     sample_rate=sample_rate / 10**9,
                                     amplitudes=voltage_amplitudes,
                                     offsets=voltage_offsets,
                                     voltage_transformations=voltage_transformation)

        segments, segment_lengths = tabor_program.get_sampled_segments()

        waveform_to_segment, to_amend, to_insert = self._find_place_for_segments_in_memory(segments,
                                                                                           segment_lengths)

        self._segment_references[waveform_to_segment[waveform_to_segment >= 0]] += 1

        for wf_index in np.flatnonzero(to_insert > 0):
            segment_index = to_insert[wf_index]
            self._upload_segment(to_insert[wf_index], segments[wf_index])
            waveform_to_segment[wf_index] = segment_index

        if np.any(to_amend):
            segments_to_amend = [segments[idx] for idx in np.flatnonzero(to_amend)]
            waveform_to_segment[to_amend] = self._amend_segments(segments_to_amend)

        self._known_programs[name] = TaborProgramMemory(waveform_to_segment=waveform_to_segment,
                                                        program=tabor_program)

    @with_configuration_guard
    @with_select
    def clear(self) -> None:
        """Delete all segments and clear memory"""
        self.device.select_channel(self._channels[0])
        self.device.send_cmd(':TRAC:DEL:ALL', paranoia_level=self.internal_paranoia_level)
        self.device.send_cmd(':SOUR:SEQ:DEL:ALL', paranoia_level=self.internal_paranoia_level)
        self.device.send_cmd(':ASEQ:DEL', paranoia_level=self.internal_paranoia_level)

        self.device.send_cmd(':TRAC:DEF 1, 192', paranoia_level=self.internal_paranoia_level)
        self.device.send_cmd(':TRAC:SEL 1', paranoia_level=self.internal_paranoia_level)
        self.device.send_cmd(':TRAC:MODE COMB', paranoia_level=self.internal_paranoia_level)
        self.device.send_binary_data(pref=':TRAC:DATA', bin_dat=self._idle_segment.get_as_binary())

        self._segment_lengths = 192*np.ones(1, dtype=np.uint32)
        self._segment_capacity = 192*np.ones(1, dtype=np.uint32)
        self._segment_hashes = np.ones(1, dtype=np.int64) * hash(self._idle_segment)
        self._segment_references = np.ones(1, dtype=np.uint32)

        self._advanced_sequence_table = []
        self._sequencer_tables = []

        self._known_programs = dict()
        self.change_armed_program(None)

    def _find_place_for_segments_in_memory(self, segments: Sequence, segment_lengths: Sequence) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        1. Find known segments
        2. Find empty spaces with fitting length
        3. Find empty spaces with bigger length
        4. Amend remaining segments
        :param segments:
        :param segment_lengths:
        :return:
        """
        segment_hashes = np.fromiter((hash(segment) for segment in segments), count=len(segments), dtype=np.int64)

        waveform_to_segment = find_positions(self._segment_hashes, segment_hashes)

        # separate into known and unknown
        unknown = (waveform_to_segment == -1)
        known = ~unknown

        known_pos_in_memory = waveform_to_segment[known]

        assert len(known_pos_in_memory) == 0 or np.all(self._segment_hashes[known_pos_in_memory] == segment_hashes[known])

        new_reference_counter = self._segment_references.copy()
        new_reference_counter[known_pos_in_memory] += 1

        to_upload_size = np.sum(segment_lengths[unknown] + 16)
        free_points_in_total = self.total_capacity - np.sum(self._segment_capacity[self._segment_references > 0])
        if free_points_in_total < to_upload_size:
            raise MemoryError('Not enough free memory',
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

            pos_of_same_length = np.logical_and(free_segments, segment_lengths[segment_idx] == self._segment_capacity[:first_free])
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
            raise MemoryError('Fragmentation does not allow upload.',
                              np.sum(segment_lengths[to_amend] + 16),
                              free_points_at_end,
                              self._free_points_at_end)

        return waveform_to_segment, to_amend, to_insert

    @with_select
    @with_configuration_guard
    def _upload_segment(self, segment_index: int, segment: TaborSegment) -> None:
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
                self.device.send_cmd(':TRAC:DEF {},{}'.format(i+1, segment_lengths[i]),
                                     paranoia_level=self.internal_paranoia_level)

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
        new_end = reserved_indices[-1]+1 if len(reserved_indices) else 0
        self._segment_lengths = self._segment_lengths[:new_end]
        self._segment_capacity = self._segment_capacity[:new_end]
        self._segment_hashes = self._segment_hashes[:new_end]
        self._segment_references = self._segment_references[:new_end]

        try:
            #  send max 10 commands at once
            chunk_size = 10
            for chunk_start in range(new_end, old_end, chunk_size):
                self.device.send_cmd('; '.join('TRAC:DEL {}'.format(i+1)
                                               for i in range(chunk_start, min(chunk_start+chunk_size, old_end))),
                                     paranoia_level=self.internal_paranoia_level)
        except Exception as e:
            raise TaborUndefinedState('Error during cleanup. Device is in undefined state.', device=self) from e

    def remove(self, name: str) -> None:
        """Remove a program from the AWG.

        Also discards all waveforms referenced only by the program identified by name.

        Args:
            name (str): The name of the program to remove.
        """
        self.free_program(name)
        self.cleanup()

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
        _ = self.device.main_instrument.send_query('*OPC?')

    def set_marker_state(self, marker: int, active: bool) -> None:
        """Sets the marker state of this channel pair.
        According to the manual one cannot turn them off/on separately."""
        command_string = ':INST:SEL {channel}; :SOUR:MARK:SEL {marker}; :SOUR:MARK:SOUR USER; :SOUR:MARK:STAT {active}'
        command_string = command_string.format(
            channel=self._channels[0],
            marker=(1, 2)[marker],
            active='ON' if active else 'OFF')
        self.device.send_cmd(command_string, paranoia_level=self.internal_paranoia_level)

    def set_channel_state(self, channel, active) -> None:
        command_string = ':INST:SEL {}; :OUTP {}'.format(self._channels[channel], 'ON' if active else 'OFF')
        self.device.send_cmd(command_string, paranoia_level=self.internal_paranoia_level)

    @with_select
    def arm(self, name: str) -> None:
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
        if name is None:
            sequencer_tables = [self._idle_sequence_table]
            advanced_sequencer_table = [(1, 1, 0)]
        else:
            waveform_to_segment_index, program = self._known_programs[name]
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

                while len(sequencer_tables[1]) < self.device.dev_properties['min_seq_len']:
                    assert advanced_sequencer_table[0][0] == 1
                    sequencer_tables[1].append((1, 1, 0))

        # insert idle sequence in advanced sequence table
        advanced_sequencer_table = [(1, 1, 1)] + advanced_sequencer_table

        while len(advanced_sequencer_table) < self.device.dev_properties['min_aseq_len']:
            advanced_sequencer_table.append((1, 1, 0))

        # reset sequencer and advanced sequencer tables to fix bug which occurs when switching between some programs
        self.device.send_cmd('SEQ:DEL:ALL', paranoia_level=self.internal_paranoia_level)
        self._sequencer_tables = []
        self.device.send_cmd('ASEQ:DEL', paranoia_level=self.internal_paranoia_level)
        self._advanced_sequence_table = []

        # download all sequence tables
        for i, sequencer_table in enumerate(sequencer_tables):
            self.device.send_cmd('SEQ:SEL {}'.format(i+1), paranoia_level=self.internal_paranoia_level)
            self.device.download_sequencer_table(sequencer_table)
        self._sequencer_tables = sequencer_tables
        self.device.send_cmd('SEQ:SEL 1', paranoia_level=self.internal_paranoia_level)

        self.device.download_adv_seq_table(advanced_sequencer_table)
        self._advanced_sequence_table = advanced_sequencer_table

        self._current_program = name

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
        return self.device.sample_rate(self._channels[0])

    @property
    def num_channels(self) -> int:
        return 2

    @property
    def num_markers(self) -> int:
        return 2

    def _enter_config_mode(self) -> None:
        """Enter the configuration mode if not already in. All outputs are set to the DC offset of the device and the
        sequencing is disabled. The manual states this speeds up sequence validation when uploading multiple sequences.
        When entering and leaving the configuration mode the AWG outputs a small (~60 mV in 4 V mode) blip."""
        if self._is_in_config_mode is False:

            # 1. Select channel pair
            # 2. Select DC as function shape
            # 3. Select build-in waveform mode

            if self.device.is_coupled():
                out_cmd = ':OUTP:ALL OFF'
            else:
                out_cmd = ':INST:SEL {};:OUTP OFF;:INST:SEL {};:OUTP OFF'.format(*self._channels)

            marker_0_cmd = ':SOUR:MARK:SEL 1;:SOUR:MARK:SOUR USER;:SOUR:MARK:STAT OFF'
            marker_1_cmd = ':SOUR:MARK:SEL 2;:SOUR:MARK:SOUR USER;:SOUR:MARK:STAT OFF'

            wf_mode_cmd = ':SOUR:FUNC:MODE FIX'

            cmd = ';'.join([out_cmd, marker_0_cmd, marker_1_cmd, wf_mode_cmd])
            self.device.send_cmd(cmd, paranoia_level=self.CONFIG_MODE_PARANOIA_LEVEL)
            self._is_in_config_mode = True

    def _exit_config_mode(self) -> None:
        """Leave the configuration mode. Enter advanced sequence mode and turn on all outputs"""

        sel_ch = ':INST:SEL {}'.format(self._channels[0])
        aseq_cmd = ':SOUR:FUNC:MODE ASEQ;SEQ:SEL 1'

        cmds = [sel_ch, aseq_cmd]

        if self.device.is_coupled():
            # Coupled -> switch all channels at once
            if self._channels == (1, 2):
                other_channel_pair = self.device.channel_pair_CD
            else:
                assert self._channels == (3, 4)
                other_channel_pair = self.device.channel_pair_AB

            if not other_channel_pair._is_in_config_mode:
                cmds.append(':OUTP:ALL ON')

        else:
            # ch 0 already selected
            cmds.append(':OUTP ON; :INST:SEL {}; :OUTP ON'.format(self._channels[1]))

        cmds.append(':SOUR:MARK:SEL 1;:SOUR:MARK:SOUR USER;:SOUR:MARK:STAT ON')
        cmds.append(':SOUR:MARK:SEL 2;:SOUR:MARK:SOUR USER;:SOUR:MARK:STAT ON')
        cmd = ';'.join(cmds)
        self.device.send_cmd(cmd, paranoia_level=self.CONFIG_MODE_PARANOIA_LEVEL)
        self._is_in_config_mode = False


class TaborUndefinedState(TaborException):
    """If this exception is raised the attached tabor device is in an undefined state.
    It is highly recommended to call reset it."""

    def __init__(self, *args, device: Union[TaborAWGRepresentation, TaborChannelPair]):
        super().__init__(*args)
        self.device = device

    def reset_device(self):
        if isinstance(self.device, TaborAWGRepresentation):
            self.device.reset()
        elif isinstance(self.device, TaborChannelPair):
            self.device.clear()
