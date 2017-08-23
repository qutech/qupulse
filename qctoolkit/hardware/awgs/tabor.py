""""""
import fractions
import sys
from typing import List, Tuple, Set, NamedTuple, Callable, Optional, Any, Sequence
from enum import Enum

# Provided by Tabor electronics for python 2.7
# a python 3 version is in a private repository on https://git.rwth-aachen.de/qutech
# Beware of the string encoding change!
import teawg
import numpy as np

from qctoolkit import ChannelID
from qctoolkit.pulses.multi_channel_pulse_template import MultiChannelWaveform
from qctoolkit.hardware.program import Loop
from qctoolkit.hardware.util import voltage_to_uint16, make_combined_wave
from qctoolkit.hardware.awgs.base import AWG


assert(sys.byteorder == 'little')


__all__ = ['TaborAWGRepresentation', 'TaborChannelPair']


class TaborSegment(tuple):
    """Represents one segment of two channels on the device. Convenience class."""
    def __new__(cls, ch_a: Optional[np.ndarray], ch_b: Optional[np.ndarray]):
        return tuple.__new__(cls, (ch_a, ch_b))

    def __init__(self, ch_a, ch_b):
        if ch_a is None and ch_b is None:
            raise TaborException('Empty TaborSegments are not allowed')
        if ch_a is not None and ch_b is not None and len(ch_a) != len(ch_b):
            raise TaborException('Channel entries to have to have the same length')

    def __hash__(self) -> int:
        return hash((bytes(self[0]) if self[0] is not None else 0,
                     bytes(self[1]) if self[1] is not None else 0))

    @property
    def num_points(self) -> int:
        return len(self[0]) if self[1] is None else len(self[1])

    def get_as_binary(self) -> np.ndarray:
        assert not (self[0] is None or self[1] is None)
        return make_combined_wave([self])


class TaborSequencing(Enum):
    SINGLE = 1
    ADVANCED = 2


class TaborProgram:
    def __init__(self,
                 program: Loop,
                 device_properties,
                 channels: Tuple[Optional[ChannelID], Optional[ChannelID]],
                 markers: Tuple[Optional[ChannelID], Optional[ChannelID]]):
        if len(channels) != device_properties['chan_per_part']:
            raise TaborException('TaborProgram only supports {} channels'.format(device_properties['chan_per_part']))
        if len(markers) != device_properties['chan_per_part']:
            raise TaborException('TaborProgram only supports {} markers'.format(device_properties['chan_per_part']))
        channel_set = frozenset(channel for channel in channels if channel is not None) | frozenset(marker
                                                                                                    for marker in
                                                                                                    markers if marker is not None)
        self._program = program

        self.__waveform_mode = None
        self._channels = tuple(channels)
        self._markers = tuple(markers)
        self.__used_channels = channel_set
        self.__device_properties = device_properties

        self._waveforms = []  # type: List[MultiChannelWaveform]
        self._sequencer_tables = []
        self._advanced_sequencer_table = []

        if self.program.repetition_count > 1:
            self.program.encapsulate()

        if self.program.depth() > 1:
            self.setup_advanced_sequence_mode()
            self.__waveform_mode = TaborSequencing.ADVANCED
        else:
            if self.program.depth() == 0:
                self.program.encapsulate()
            self.setup_single_sequence_mode()
            self.__waveform_mode = TaborSequencing.SINGLE

    @property
    def markers(self) -> Tuple[Optional[ChannelID], Optional[ChannelID]]:
        return self._markers

    @property
    def channels(self) -> Tuple[Optional[ChannelID], Optional[ChannelID]]:
        return self._channels

    def sampled_segments(self,
                         sample_rate: float,
                         voltage_amplitude: Tuple[float, float],
                         voltage_offset: Tuple[float, float],
                         voltage_transformation: Tuple[Callable, Callable]) -> Tuple[Sequence[TaborSegment],
                                                                                     Sequence[int]]:
        sample_rate = fractions.Fraction(sample_rate, 10**9)

        segment_lengths = [waveform.duration*sample_rate for waveform in self._waveforms]
        if not all(abs(int(segment_length) - segment_length) < 1e-10 and segment_length > 0
                   for segment_length in segment_lengths):
            raise TaborException('At least one waveform has a length that is no integer or smaller zero')
        segment_lengths = np.asarray(segment_lengths, dtype=np.uint64)

        if np.any(segment_lengths % 16 > 0) or np.any(segment_lengths < 192):
            raise TaborException('At least one waveform has a length that is smaller 192 or not a multiple of 16')
        sample_rate = float(sample_rate)
        time_array = np.arange(np.max(segment_lengths)) / sample_rate

        def voltage_to_data(waveform, time, channel):
            if self._channels[channel]:
                return voltage_to_uint16(
                    voltage_transformation[channel](
                        waveform.get_sampled(channel=self._channels[channel],
                                             sample_times=time)),
                    voltage_amplitude[channel],
                    voltage_offset[channel],
                    resolution=14)
            else:
                return np.zeros(len(time), dtype=np.uint16)

        def get_marker_data(waveform: MultiChannelWaveform, time):
            marker_data = np.zeros(len(time), dtype=np.uint16)
            for marker_index, markerID in enumerate(self._markers):
                if markerID is not None:
                    marker_data |= (waveform.get_sampled(channel=markerID, sample_times=time) != 0).\
                                       astype(dtype=np.uint16) << marker_index+14
            return marker_data

        segments = np.empty_like(self._waveforms, dtype=TaborSegment)
        for i, waveform in enumerate(self._waveforms):
            t = time_array[:int(waveform.duration*sample_rate)]
            segment_a = voltage_to_data(waveform, t, 0)
            segment_b = voltage_to_data(waveform, t, 1)
            assert (len(segment_a) == len(t))
            assert (len(segment_b) == len(t))
            seg_data = get_marker_data(waveform, t)
            segment_a |= seg_data
            segments[i] = TaborSegment(segment_a, segment_b)
        return segments, segment_lengths

    def setup_single_sequence_mode(self) -> None:
        assert self.program.depth() == 1

        sequencer_table = []
        waveforms = []

        for waveform, repetition_count in ((waveform_loop.waveform.get_subset_for_channels(self.__used_channels),
                                            waveform_loop.repetition_count)
                                           for waveform_loop in self.program):
            if waveform in waveforms:
                waveform_index = waveforms.index(waveform)
            else:
                waveform_index = len(waveforms)
                waveforms.append(waveform)
            sequencer_table.append((repetition_count, waveform_index, 0))

        self._waveforms = waveforms
        self._sequencer_tables = [sequencer_table]
        self._advanced_sequencer_table = [(self.program.repetition_count, 1, 0)]

    def setup_advanced_sequence_mode(self) -> None:
        assert self.program.depth() > 1
        assert self.program.repetition_count == 1

        self.program.flatten_and_balance(2)

        min_seq_len = self.__device_properties['min_seq_len']
        max_seq_len = self.__device_properties['max_seq_len']

        def check_merge_with_next(program, n):
            if (program[n].repetition_count == 1 and program[n+1].repetition_count == 1 and
                    len(program[n]) + len(program[n+1]) < max_seq_len):
                program[n][len(program[n]):] = program[n + 1][:]
                program[n + 1:n + 2] = []
                return True
            return False

        def check_partial_unroll(program, n):
            st = program[n]
            if sum(entry.repetition_count for entry in st) * st.repetition_count >= min_seq_len:
                if sum(entry.repetition_count for entry in st) < min_seq_len:
                    st.unroll_children()
                while len(st) < min_seq_len:
                    st.split_one_child()
                return True
            return False

        i = 0
        while i < len(self.program):
            self.program[i].assert_tree_integrity()
            if len(self.program[i]) > max_seq_len:
                raise TaborException('The algorithm is not smart enough to make sequence tables shorter')
            elif len(self.program[i]) < min_seq_len:
                assert self.program[i].repetition_count > 0
                if self.program[i].repetition_count == 1:
                    # check if merging with neighbour is possible
                    if i > 0 and check_merge_with_next(self.program, i-1):
                        pass
                    elif i+1 < len(self.program) and check_merge_with_next(self.program, i):
                        pass

                    # check if (partial) unrolling is possible
                    elif check_partial_unroll(self.program, i):
                        i += 1

                    # check if sequence table can be extended by unrolling a neighbor
                    elif (i > 0
                          and self.program[i - 1].repetition_count > 1
                          and len(self.program[i]) + len(self.program[i-1]) < max_seq_len):
                        self.program[i][:0] = self.program[i-1].copy_tree_structure()[:]
                        self.program[i - 1].repetition_count -= 1

                    elif (i+1 < len(self.program)
                          and self.program[i+1].repetition_count > 1
                          and len(self.program[i]) + len(self.program[i+1]) < max_seq_len):
                        self.program[i][len(self.program[i]):] = self.program[i+1].copy_tree_structure()[:]
                        self.program[i+1].repetition_count -= 1

                    else:
                        raise TaborException('The algorithm is not smart enough to make this sequence table longer')
                elif check_partial_unroll(self.program, i):
                    i += 1
                else:
                    raise TaborException('The algorithm is not smart enough to make this sequence table longer')
            else:
                i += 1

        for sequence_table in self.program:
            assert len(sequence_table) >= self.__device_properties['min_seq_len']
            assert len(sequence_table) <= self.__device_properties['max_seq_len']

        advanced_sequencer_table = []
        sequencer_tables = []
        waveforms = []
        for sequencer_table_loop in self.program:
            current_sequencer_table = []
            for waveform, repetition_count in ((waveform_loop.waveform.get_subset_for_channels(self.__used_channels),
                                                waveform_loop.repetition_count)
                                               for waveform_loop in sequencer_table_loop):
                if waveform in waveforms:
                    wf_index = waveforms.index(waveform)
                else:
                    wf_index = len(waveforms)
                    waveforms.append(waveform)
                current_sequencer_table.append((repetition_count, wf_index, 0))

            if current_sequencer_table in sequencer_tables:
                sequence_no = sequencer_tables.index(current_sequencer_table) + 1
            else:
                sequence_no = len(sequencer_tables) + 1
                sequencer_tables.append(current_sequencer_table)

            advanced_sequencer_table.append((sequencer_table_loop.repetition_count, sequence_no, 0))

        self._advanced_sequencer_table = advanced_sequencer_table
        self._sequencer_tables = sequencer_tables
        self._waveforms = waveforms

    @property
    def program(self) -> Loop:
        return self._program

    def get_sequencer_tables(self) -> List[Tuple[int, int, int]]:
        return self._sequencer_tables

    def get_advanced_sequencer_table(self) -> List[Tuple[int, int, int]]:
        """Advanced sequencer table that can be used  via the download_adv_seq_table pytabor command"""
        return self._advanced_sequencer_table

    @property
    def waveform_mode(self) -> str:
        return self.__waveform_mode


class TaborAWGRepresentation(teawg.TEWXAwg):
    def __init__(self, *args, external_trigger=False, reset=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._clock_marker = [0, 0, 0, 0]

        if reset:
            self.visa_inst.write(':RES')

        if external_trigger:
            raise NotImplementedError()  # pragma: no cover

        def switch_group_off(grp):
            switch_off_cmd = (":INST:SEL {}; :OUTP OFF; :INST:SEL {}; :OUTP OFF; "
                              ":SOUR:MARK:SEL 1; :SOUR:MARK:STAT OFF; :SOUR:MARK:SOUR USER; "
                              ":SOUR:MARK:SEL 2; :SOUR:MARK:STAT OFF; :SOUR:MARK:SOUR USER").format(grp+1, grp+2)
            self.send_cmd(switch_off_cmd)
        switch_group_off(0)
        switch_group_off(1)

        setup_command = (
                         ":INIT:GATE OFF; :INIT:CONT ON; "
                         ":INIT:CONT:ENAB ARM; :INIT:CONT:ENAB:SOUR BUS; "
                         ":SOUR:SEQ:JUMP:EVEN BUS")
        # Set Software Trigger Mode
        self.select_channel(1)
        self.send_cmd(setup_command)
        self.select_channel(3)
        self.send_cmd(setup_command)

    def send_cmd(self, cmd_str, paranoia_level=None) -> Any:
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

    @property
    def is_open(self) -> bool:
        return self.visa_inst is not None  # pragma: no cover

    def select_channel(self, channel) -> None:
        if channel not in (1, 2, 3, 4):
            raise TaborException('Invalid channel: {}'.format(channel))

        self.send_cmd(':INST:SEL {channel}'.format(channel=channel))

    def select_marker(self, marker) -> None:
        if marker not in (1, 2, 3, 4):
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

    def offset(self, channel) -> float:
        return float(self.send_query(':INST:SEL {channel}; :VOLT:OFFS?'.format(channel=channel)))

    def enable(self) -> None:
        self.send_cmd(':ENAB')

    def abort(self) -> None:
        self.send_cmd(':ABOR')

    def reset(self) -> None:
        self.send_cmd(':RES')
        self.send_cmd(':INST:SEL 1; :INIT:GATE OFF; :INIT:CONT ON; :INIT:CONT:ENAB ARM; :INIT:CONT:ENAB:SOUR BUS')
        self.send_cmd(':INST:SEL 3; :INIT:GATE OFF; :INIT:CONT ON; :INIT:CONT:ENAB ARM; :INIT:CONT:ENAB:SOUR BUS')

    def trigger(self) -> None:
        self.send_cmd(':TRIG')


TaborProgramMemory = NamedTuple('TaborProgramMemory', [('segment_indices', np.ndarray),
                                                       ('program', TaborProgram)])


def with_configuration_guard(function_object: Callable[['TaborChannelPair'], Any]) -> Callable[['TaborChannelPair'],
                                                                                               Any]:
    """This decorator assures that the AWG is in configuration mode while the decorated method runs."""
    def guarding_method(channel_pair: 'TaborChannelPair', *args, **kwargs) -> Any:
        if channel_pair._configuration_guard_count == 0:
            channel_pair._enter_config_mode()
        channel_pair._configuration_guard_count += 1

        function_object(channel_pair, *args, **kwargs)

        channel_pair._configuration_guard_count -= 1
        if channel_pair._configuration_guard_count == 0:
            channel_pair._exit_config_mode()

    return guarding_method


class TaborChannelPair(AWG):
    def __init__(self, tabor_device: TaborAWGRepresentation, channels: Tuple[int, int], identifier: str):
        super().__init__(identifier)
        self._device = tabor_device

        self._configuration_guard_count = 0
        self._is_in_config_mode = False

        if channels not in ((1, 2), (3, 4)):
            raise ValueError('Invalid channel pair: {}'.format(channels))
        self._channels = channels

        self._enter_config_mode()

        # Select channel 1
        self._device.select_channel(self._channels[0])
        self._device.send_cmd(':TRAC:DEL:ALL')
        self._device.send_cmd(':SOUR:SEQ:DEL:ALL')
        self._device.send_cmd(':ASEQ:DEL')

        self._idle_segment = TaborSegment(voltage_to_uint16(voltage=np.zeros(192),
                                                            output_amplitude=0.5,
                                                            output_offset=0., resolution=14),
                                          voltage_to_uint16(voltage=np.zeros(192),
                                                            output_amplitude=0.5,
                                                            output_offset=0., resolution=14))
        self._idle_sequence_table = [(1, 1, 0), (1, 1, 0), (1, 1, 0)]

        self._device.send_cmd(':TRAC:DEF 1, 192')
        self._device.send_cmd(':TRAC:SEL 1')
        self._device.send_cmd(':TRAC:MODE COMB')
        self._device.send_binary_data(pref=':TRAC:DATA', bin_dat=self._idle_segment.get_as_binary())

        self._segment_lengths = 192*np.ones(1, dtype=np.uint32)
        self._segment_capacity = 192*np.ones(1, dtype=np.uint32)
        self._segment_hashes = np.ones(1, dtype=np.int64) * hash(self._idle_segment)
        self._segment_references = np.ones(1, dtype=np.uint32)

        self._device.send_cmd('SEQ:SEL 1')
        self._device.download_sequencer_table(self._idle_sequence_table)
        self._sequencer_tables = [self._idle_sequence_table]

        self._advanced_sequence_table = []

        self.total_capacity = int(16e6)

        self._known_programs = dict()  # type: Dict[str, TaborProgramMemory]
        self._current_program = None

        self._exit_config_mode()

    def free_program(self, name: str) -> TaborProgramMemory:
        program = self._known_programs.pop(name)
        self._segment_references[program.segment_indices] -= 1
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
        if reserved_index:
            return self.total_capacity - np.sum(self._segment_capacity[:reserved_index[-1]])
        else:
            return self.total_capacity

    @with_configuration_guard
    def upload(self, name: str,
               program: Loop,
               channels: List[ChannelID],
               markers: List[ChannelID],
               voltage_transformation: List[Callable],
               force: bool=False) -> None:
        """Upload a program to the AWG.

        The policy is to prefer amending the unknown waveforms to overwriting old ones."""

        if len(channels) != self.num_channels:
            raise ValueError('Channel ID not specified')
        if len(markers) != self.num_markers:
            raise ValueError('Markers not specified')
        if len(voltage_transformation) != self.num_channels:
            raise ValueError('Wrong number of voltage transformations')

        # helper to restore previous state if upload is impossible
        to_restore = None
        if name in self._known_programs:
            if force:
                # save old program to restore in on error
                to_restore = self.free_program(name)
            else:
                raise ValueError('{} is already known on {}'.format(name, self.identifier))

        try:
            # parse to tabor program
            tabor_program = TaborProgram(program,
                                         channels=tuple(channels),
                                         markers=markers,
                                         device_properties=self._device.dev_properties)
            sample_rate = self._device.sample_rate(self._channels[0])
            voltage_amplitudes = (self._device.amplitude(self._channels[0]),
                                  self._device.amplitude(self._channels[1]))
            voltage_offsets = (self._device.offset(self._channels[0]),
                               self._device.offset(self._channels[1]))
            segments, segment_lengths = tabor_program.sampled_segments(sample_rate=sample_rate,
                                                                       voltage_amplitude=voltage_amplitudes,
                                                                       voltage_offset=voltage_offsets,
                                                                       voltage_transformation=voltage_transformation)
            segment_hashes = np.fromiter((hash(segment) for segment in segments), count=len(segments), dtype=np.uint64)

            known_waveforms = np.in1d(segment_hashes, self._segment_hashes, assume_unique=True)
            to_upload_size = np.sum(segment_lengths[~known_waveforms] + 16)

            waveform_to_segment = np.full(len(segments), -1, dtype=int)
            waveform_to_segment[known_waveforms] = np.flatnonzero(
                np.in1d(self._segment_hashes, segment_hashes[known_waveforms]))

            to_amend = ~known_waveforms
            to_insert = []

            if name not in self._known_programs:
                if self._free_points_in_total < to_upload_size:
                    raise MemoryError('Not enough free memory')
                if self._free_points_at_end < to_upload_size:
                    reserved_indices = np.flatnonzero(self._segment_reserved)
                    if len(reserved_indices) == 0:
                        raise MemoryError('Fragmentation does not allow upload.')

                    last_reserved = reserved_indices[-1] if reserved_indices else 0
                    free_segments = np.flatnonzero(self._segment_references[:last_reserved] == 0)[
                        np.argsort(self._segment_capacity[:last_reserved])[::-1]]

                    for wf_index in np.argsort(segment_lengths[~known_waveforms])[::-1]:
                        if segment_lengths[wf_index] <= self._segment_capacity[free_segments[0]]:
                            to_insert.append((wf_index, free_segments[0]))
                            free_segments = free_segments[1:]
                            to_amend[wf_index] = False

                    if np.sum(segment_lengths[to_amend] + 16) > self._free_points_at_end:
                        raise MemoryError('Fragmentation does not allow upload.')

        except:
            if to_restore:
                self._known_programs[name] = to_restore
                self._segment_reserved[to_restore.segment_indices] += 1
            raise

        self._segment_references[waveform_to_segment[waveform_to_segment >= 0]] += 1

        if to_insert:
            for wf_index, segment_index in to_insert:
                self._upload_segment(segment_index, segments[wf_index])
                waveform_to_segment[wf_index] = segment_index

        if np.any(to_amend):
            segments_to_amend = segments[to_amend]
            self._amend_segments(segments_to_amend)
            waveform_to_segment[to_amend] = np.arange(len(self._segment_capacity) - np.sum(to_amend),
                                                      len(self._segment_capacity), dtype=int) + 1

        self._known_programs[name] = TaborProgramMemory(segment_indices=waveform_to_segment,
                                                        program=tabor_program)

    @with_configuration_guard
    def _upload_segment(self, segment_index: int, segment: TaborSegment) -> None:
        if self._segment_references[segment_index] > 0:
            raise ValueError('Reference count not zero')
        if segment.num_points > self._segment_capacity[segment_index]:
            raise ValueError('Cannot upload segment here.')


        self._device.send_cmd(':TRAC:DEF {}, {}'.format(segment_index, segment.num_points))
        self._segment_lengths[segment_index] = segment.num_points

        self._device.send_cmd(':TRAC:SEL {}'.format(segment_index))

        self._device.send_cmd('TRAC:MODE COMB')
        wf_data = segment.get_as_binary()

        self._device.send_binary_data(pref=':TRAC:DATA', bin_dat=wf_data)
        self._segment_references[segment_index] = 1
        self._segment_hashes[segment_index] = hash(segment)

    @with_configuration_guard
    def _amend_segments(self, segments: List[TaborSegment]) -> None:
        new_lengths = np.asarray([s.num_points for s in segments], dtype=np.uint32)

        wf_data = make_combined_wave(segments)
        trac_len = len(wf_data) // 2

        segment_index = len(self._segment_capacity) + 1
        self._device.send_cmd(':TRAC:DEF {}, {}'.format(segment_index, trac_len))
        self._device.send_cmd(':TRAC:SEL {}'.format(segment_index))
        self._device.send_cmd('TRAC:MODE COMB')
        self._device.send_binary_data(pref=':TRAC:DATA', bin_dat=wf_data)

        old_to_update = np.count_nonzero(self._segment_capacity != self._segment_lengths and self._segment_reserved)
        segment_capacity = np.concatenate((self._segment_capacity, new_lengths))
        segment_lengths = np.concatenate((self._segment_lengths, new_lengths))
        segment_references = np.concatenate((self._segment_references, np.ones(len(segments), dtype=int)))
        segment_hashes = np.concatenate((self._segment_hashes, [hash(s) for s in segments]))
        if len(segments) < old_to_update:
            for i, segment in enumerate(segments):
                current_segment = segment_index + i
                self._device.send_cmd(':TRAC:DEF {}, {}'.format(current_segment, segment.num_points))
        else:
            # flush the capacity
            self._device.download_segment_lengths(segment_capacity)

            # update non fitting lengths
            for i in np.flatnonzero(np.logical_and(segment_capacity != segment_lengths, segment_references > 0)):
                self._device.send_cmd(':TRAC:DEF {},{}'.format(i, segment_lengths[i]))

        self._segment_capacity = segment_capacity
        self._segment_lengths = segment_lengths
        self._segment_hashes = segment_hashes
        self._segment_references = segment_references

    def cleanup(self) -> None:
        """Discard all segments after the last which is still referenced"""
        reserved_indices = np.flatnonzero(self._segment_references > 0)

        new_end = reserved_indices[-1]+1 if reserved_indices else 0
        self._segment_lengths = self._segment_lengths[:new_end]
        self._segment_capacity = self._segment_capacity[:new_end]
        self._segment_hashes = self._segment_capacity[:new_end]
        self._segment_references = self._segment_capacity[:new_end]

    def remove(self, name: str) -> None:
        """Remove a program from the AWG.

        Also discards all waveforms referenced only by the program identified by name.

        Args:
            name (str): The name of the program to remove.
        """
        self.free_program(name)
        self.cleanup()

    def set_marker_state(self, marker, active) -> None:
        command_string = ':INST:SEL {}; :SOUR:MARK:SEL {}; :SOUR:MARK:SOUR USER; :SOUR:MARK:STAT {}'.format(
            self._channels[0],
            marker+1,
            'ON' if active else 'OFF')
        self._device.send_cmd(command_string)

    def set_channel_state(self, channel, active) -> None:
        command_string = ':INST:SEL {}; :OUTP {}'.format(self._channels[channel], 'ON' if active else 'OFF')
        self._device.send_cmd(command_string)

    def arm(self, name: str) -> None:
        if self._current_program == name:
            self._device.send_cmd('SEQ:SEL 1')
        else:
            self.change_armed_program(name)

    @with_configuration_guard
    def change_armed_program(self, name: str) -> None:
        waveform_to_segment, program = self._known_programs[name]

        # translate waveform number to actual segment
        sequencer_tables = [[(rep_count, waveform_to_segment[wf_index-1], jump_flag)
                             for (rep_count, wf_index, jump_flag) in sequencer_table]
                            for sequencer_table in program.get_sequencer_tables()]

        # insert idle sequence
        sequencer_tables = [self._idle_sequence_table] + sequencer_tables

        # adjust advanced sequence table entries by idle sequence table offset
        advanced_sequencer_table = [(rep_count, seq_no + 1, jump_flag)
                                    for rep_count, seq_no, jump_flag in program.get_advanced_sequencer_table()]

        if program.waveform_mode == TaborSequencing.SINGLE:
            assert len(advanced_sequencer_table) == 1
            assert len(sequencer_tables) == 2

            while len(sequencer_tables[1]) < self._device.dev_properties['min_seq_len']:
                sequencer_tables[1].append((1, 1, 0))

        # insert idle sequence in advanced sequence table
        advanced_sequencer_table = [(1, 1, 1)] + advanced_sequencer_table

        while len(advanced_sequencer_table) < self._device.dev_properties['min_aseq_len']:
            advanced_sequencer_table.append((1, 1, 0))

        #download all sequence tables
        for i, sequencer_table in enumerate(sequencer_tables):
            if i >= len(self._sequencer_tables) or self._sequencer_tables[i] != sequencer_table:
                self._device.send_cmd('SEQ:SEL {}'.format(i+1))
                self._device.download_sequencer_table(sequencer_table)
        self._sequencer_tables = sequencer_tables
        self._device.send_cmd('SEQ:SEL 1')

        self._device.download_adv_seq_table(advanced_sequencer_table)
        self._advanced_sequence_table = advanced_sequencer_table

        # this will set the DC voltage to the first value of the idle waveform
        self._device.enable()
        self._current_program = name

    def run_current_program(self) -> None:
        if self._current_program:
            self._device.send_cmd(':TRIG')
        else:
            raise RuntimeError('No program active')

    @property
    def programs(self) -> Set[str]:
        """The set of program names that can currently be executed on the hardware AWG."""
        raise set(program.name for program in self._known_programs.keys())

    @property
    def sample_rate(self) -> float:
        return self._device.sample_rate(self._channels[0])

    @property
    def num_channels(self) -> int:
        return 2

    @property
    def num_markers(self) -> int:
        return 2

    def _enter_config_mode(self) -> None:
        """Enter the configuration mode if not already in. All outputs are turned of and the sequencing is disabled
        as the manual states this speeds up sequence validation when uploading multiple sequences"""
        if self._is_in_config_mode is False:
            self.set_marker_state(0, False)
            self.set_marker_state(1, False)

            self.set_channel_state(0, False)
            self.set_channel_state(1, False)

            self._device.abort()

            self._device.send_cmd(':SOUR:FUNC:MODE FIX')
            self._is_in_config_mode = True

    def _exit_config_mode(self) -> None:
        """Leave the configuration mode. Enter advanced sequence mode and turn on all outputs"""
        if self._current_program:
            _, program = self._known_programs[self._current_program]

            self._device.send_cmd(':SOUR:FUNC:MODE ASEQ')

            self.set_marker_state(0, program.markers[0] is not None)
            self.set_marker_state(1, program.markers[1] is not None)

            self.set_channel_state(0, program.channels[0] is not None)
            self.set_channel_state(1, program.channels[1] is not None)

            self._is_in_config_mode = False




class TaborException(Exception):
    pass
