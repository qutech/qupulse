from qctoolkit.pulses.pulse_template import ChannelID
from qctoolkit.pulses.multi_channel_pulse_template import MultiChannelWaveform
from qctoolkit.hardware.program import Loop, MultiChannelProgram
from qctoolkit.hardware.util import voltage_to_uint16
from qctoolkit.hardware.awgs import AWG

import sys
import numpy as np
from typing import List, Tuple, Iterable, Set, NamedTuple

# Provided by Tabor electronics for python 2.7
# a python 3 version is in a private repository on https://git.rwth-aachen.de/qutech
# Beware of the string encoding change!
import pytabor
import teawg

assert(sys.byteorder == 'little')


class TaborSegment(tuple):
    def __new__(cls, ch_a, ch_b):
        return tuple.__new__(cls, (ch_a, ch_b))

    def __init__(self, ch_a, ch_b):
        pass

    def num_points(self) -> int:
        return max(len(self[0]), len(self[1]))

    def __hash__(self) -> int:
        return hash((bytes(self[0]), bytes(self[1])))


class TaborProgram:
    WAVEFORM_MODES = ('single', 'advanced', 'sequence')

    def __init__(self,
                 program: MultiChannelProgram,
                 device_properties,
                 channels: Tuple[ChannelID, ChannelID],
                 markers: Tuple[ChannelID, ChannelID]):
        if len(channels) != device_properties['chan_per_part']:
            raise TaborException('TaborProgram only supports {} channels'.format(device_properties['chan_per_part']))
        if len(markers) != device_properties['chan_per_part']:
            raise TaborException('TaborProgram only supports {} markers'.format(device_properties['chan_per_part']))
        channel_set = frozenset(channel for channel in channels if channel is not None) | frozenset(marker
                                                                                                    for marker in
                                                                                                    markers if marker is not None)
        self.__root_loop = None
        for known_channels in program.programs.keys():
            if known_channels.issuperset(channel_set):
                self.__root_loop = program.programs[known_channels]
                break
        if self.__root_loop is None:
            raise TaborException("{} not found in program.".format(channel_set))

        self.__waveform_mode = 'advanced'
        self.__channels = channels
        self.__markers = markers
        self.__used_channels = channel_set
        self.__device_properties = device_properties

        self.__waveforms = []  # type: List[MultiChannelWaveform]
        self.__sequencer_tables = []
        self.__advanced_sequencer_table = []

        if self.program.depth() == 0:
            self.setup_single_waveform_mode()
        elif self.program.depth() == 1:
            self.setup_single_sequence_table_mode()
        else:
            self.setup_advanced_sequence_mode()

    def setup_single_waveform_mode(self) -> None:
        raise NotImplementedError()

    def sampled_segments(self,
                         sample_rate: float,
                         voltage_amplitude: Tuple[float, float],
                         voltage_offset: Tuple[float, float]) -> List[TaborSegment]:

        segment_lengths = np.fromiter((waveform.duration for waveform in self.__waveforms),
                                      dtype=float, count=len(self.__waveforms)) * sample_rate
        if not all(segment_length.is_integer() for segment_length in segment_lengths):
            raise TaborException('At least one waveform has a length that is no multiple of the time per sample')
        segment_lengths = segment_lengths.astype(dtype=int)
        time_array = np.arange(np.max(segment_lengths)) / sample_rate

        def voltage_to_data(waveform, time, channel):
            if self.__channels[channel]:
                return voltage_to_uint16(
                    waveform[self.__channels[channel]].get_sampled(channel=self.__channels[channel],
                                                                   sample_times=time),
                    voltage_amplitude[channel],
                    voltage_offset[channel],
                    resolution=14)
            else:
                return np.zeros(len(time), dtype=np.uint16)

        def get_marker_data(waveform: MultiChannelWaveform, time):
            marker_data = np.zeros(len(time), dtype=np.uint16)
            for marker_index, markerID in enumerate(self.__markers):
                if markerID is not None:
                    marker_data |= (waveform.get_sampled(channel=markerID, sample_times=time) != 0).\
                                       astype(dtype=np.uint16) << marker_index+14
            return marker_data

        segments = len(self.__waveforms)*[None]
        for i, waveform in enumerate(self.__waveforms):
            t = time_array[:int(waveform.duration*sample_rate)]
            segment_a = voltage_to_data(waveform, t, 0)
            segment_b = voltage_to_data(waveform, t, 1)
            assert (len(segment_a) == len(t))
            assert (len(segment_b) == len(t))
            seg_data = get_marker_data(waveform, t)
            segment_b |= seg_data
            segments[i] = TaborSegment(segment_a, segment_b)
        return segments, segment_lengths

    def setup_single_sequence_mode(self) -> None:
        self.__waveform_mode = 'sequence'
        if len(self.program) < self.__device_properties['min_seq_len']:
            raise TaborException('SEQuence:LENGth has to be >={min_seq_len}'.format(**self.__device_properties))
        raise NotImplementedError()

    def setup_advanced_sequence_mode(self) -> None:
        while self.program.depth() > 2 or not self.program.is_balanced():
            for i, sequence_table in enumerate(self.program):
                if sequence_table.depth() == 0:
                    sequence_table.encapsulate()
                elif sequence_table.depth() == 1:
                    assert (sequence_table.is_balanced())
                elif len(sequence_table) == 1 and len(sequence_table[0]) == 1:
                    sequence_table.join_loops()
                elif sequence_table.is_balanced():
                    if len(self.program) < self.__device_properties['min_aseq_len'] or (sequence_table.repetition_count / self.__device_properties['max_aseq_len'] <
                                                max(entry.repetition_count for entry in sequence_table) /
                                                self.__device_properties['max_seq_len']):
                        sequence_table.unroll()
                    else:
                        for entry in sequence_table.children:
                            entry.unroll()

                else:
                    depth_to_unroll = sequence_table.depth() - 1
                    for entry in sequence_table:
                        if entry.depth() == depth_to_unroll:
                            entry.unroll()

        i = 0
        while i < len(self.program):
            sequence_table = self.program[i]
            if len(sequence_table) > self.__device_properties['max_seq_len']:
                raise TaborException()
            elif len(sequence_table) < self.__device_properties['min_seq_len']:
                # try to merge with neighbours
                if sequence_table.repetition_count == 1:
                    if i > 0 and self.program[i-1].repetition_count == 1:
                        self.program[i-1][len(self.program[i-1]):] = sequence_table[:]
                        self.program[i:i+1] = []
                    elif i+1 < len(self.program) and self.program[i+1].repetition_count == 1:
                        self.program[i+1][:0] = sequence_table[:]
                        self.program[i:i+1] = []
                    else:
                        self.increase_sequence_table_length(sequence_table, self.__device_properties)
                        i += 1
                else:
                    self.increase_sequence_table_length(sequence_table, self.__device_properties)
                    i += 1
            else:
                i += 1

        assert (self.program.repetition_count == 1)
        if len(self.program) < self.__device_properties['min_aseq_len']:
            raise TaborException()
        if len(self.program) > self.__device_properties['max_aseq_len']:
            raise TaborException()
        for sequence_table in self.program:
            if len(sequence_table) < self.__device_properties['min_seq_len']:
                raise TaborException()
            if len(sequence_table) > self.__device_properties['max_seq_len']:
                raise TaborException()

        advanced_sequencer_table = []
        sequencer_tables = []
        waveforms = []
        for sequencer_table_loop in self.program:
            current_sequencer_table = []
            for waveform, repetition_count in ((waveform_loop.waveform.get_subset_for_channels(self.__used_channels),
                                                waveform_loop.repetition_count)
                                               for waveform_loop in sequencer_table_loop):
                if waveform in waveforms:
                    segment_no = waveforms.index(waveform) + 1
                else:
                    segment_no = len(waveforms) + 1
                    waveforms.append(waveform)
                current_sequencer_table.append((repetition_count, segment_no, 0))

            if current_sequencer_table in sequencer_tables:
                sequence_no = sequencer_tables.index(current_sequencer_table) + 1
            else:
                sequence_no = len(sequencer_tables) + 1
                sequencer_tables.append(current_sequencer_table)

            advanced_sequencer_table.append((sequencer_table_loop.repetition_count, sequence_no, 0))

        self.__advanced_sequencer_table = advanced_sequencer_table
        self.__sequencer_tables = sequencer_tables
        self.__waveforms = waveforms

    @property
    def program(self) -> Loop:
        return self.__root_loop

    def get_sequencer_tables(self) -> List[Tuple[int, int, int]]:
        return self.__sequencer_tables

    @staticmethod
    def increase_sequence_table_length(sequence_table: Loop, device_properties) -> None:
        assert(sequence_table.depth() == 1)
        if len(sequence_table) < device_properties['min_seq_len']:

            if sum(entry.repetition_count for entry in sequence_table)*sequence_table.repetition_count >= device_properties['min_seq_len']:
                if sum(entry.repetition_count for entry in sequence_table) < device_properties['min_seq_len']:
                    sequence_table.unroll_children()
                while len(sequence_table) < device_properties['min_seq_len']:
                    sequence_table.split_one_child()
            else:
                TaborException('Sequence table too short: ', sequence_table)

    def get_advanced_sequencer_table(self) -> List[Tuple[int, int, int]]:
        """Advanced sequencer table that can be used  via the download_adv_seq_table pytabor command"""
        return self.__advanced_sequencer_table

    def get_waveform_data(self,
                          device_properties,
                          sample_rate: float,
                          voltage_amplitude: Tuple[float, float],
                          voltage_offset: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        if any(not(waveform.duration*sample_rate).is_integer() for waveform in self.__waveforms):
            raise TaborException('At least one waveform has a length that is no multiple of the time per sample')
        maximal_length = int(max(waveform.duration for waveform in self.__waveforms) * sample_rate)
        time_array = np.arange(0, maximal_length, 1)
        maximal_size = int(2 * (sample_rate*sum(waveform.duration for waveform in self.__waveforms) + 16*len(self.__waveforms)))
        data = np.empty(maximal_size, dtype=np.uint16)
        offset = 0
        segment_lengths = np.zeros(len(self.__waveforms), dtype=np.uint32)

        def voltage_to_data(waveform, time, channel):
            return voltage_to_uint16(waveform[self.__channels[channel]].sample(time),
                                     voltage_amplitude[channel],
                                     voltage_offset[channel],
                                     resolution=14) if self.__channels[channel] else None

        for i, waveform in enumerate(self.__waveforms):
            t = time_array[:int(waveform.duration*sample_rate)]
            segment1 = voltage_to_data(waveform, t, 0)
            segment2 = voltage_to_data(waveform, t, 1)
            segment_lengths[i] = len(segment1) if segment1 is not None else len(segment2)
            assert(segment2 is None or segment_lengths[i] == len(segment2))
            offset = pytabor.make_combined_wave(
                segment1,
                segment2,
                data, offset, add_idle_pts=True)

        if np.any(segment_lengths < device_properties['min_seq_len']):
            raise Exception()
        if np.any(segment_lengths % device_properties['seg_quantum']>0):
            raise Exception()

        return data[:offset], segment_lengths

    def upload_to_device(self, device: 'TaborAWG', channel_pair) -> None:
        if channel_pair not in ((1, 2), (3, 4)):
            raise Exception('Invalid channel pair', channel_pair)

        if self.__waveform_mode == 'advanced':
            sample_rate = device.sample_rate(channel_pair[0])
            amplitude = (device.amplitude(channel_pair[0]), device.amplitude(channel_pair[1]))
            offset = (device.offset(channel_pair[0]), device.offset(channel_pair[1]))

            wf_data, segment_lengths = self.get_waveform_data(device_properties=device.dev_properties,
                                                              sample_rate=sample_rate,
                                                              voltage_amplitude=amplitude,
                                                              voltage_offset=offset)
            # download the waveform data as one big waveform
            device.select_channel(channel_pair[0])
            device.send_cmd(':FUNC:MODE ASEQ')
            device.send_cmd(':TRAC:DEF 1,{}'.format(len(wf_data)))
            device.send_cmd(':TRAC:SEL 1')
            device.send_binary_data(pref=':TRAC:DATA', bin_dat=wf_data)
            # partition the memory
            device.download_segment_lengths(segment_lengths)

            #download all sequence tables
            for i, sequencer_table in enumerate(self.get_sequencer_tables()):
                device.send_cmd('SEQ:SEL {}'.format(i+1))
                device.download_sequencer_table(sequencer_table)

            device.download_adv_seq_table(self.get_advanced_sequencer_table())
            device.send_cmd('SEQ:SEL 1')


class TaborAWGRepresentation(teawg.TEWXAwg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__selected_channel = None

    @property
    def is_open(self) -> bool:
        return self.visa_inst is not None

    def select_channel(self, channel) -> None:
        if channel not in (1, 2, 3, 4):
            raise TaborException('Invalid channel: {}'.format(channel))
        if self.__selected_channel != channel:
            self.send_cmd(':INST:SEL {channel}'.format(channel=channel))
            self.__selected_channel = channel

    def sample_rate(self, channel) -> int:
        self.select_channel(channel)
        return int(float(self.send_query(':FREQ:RAST?'.format(channel=channel))))

    def amplitude(self, channel) -> float:
        self.select_channel(channel)
        coupling = self.send_query(':OUTP:COUP?'.format(channel=channel))
        if coupling == 'DC':
            return float(self.send_query(':VOLT?'))
        elif coupling == 'HV':
            return float(self.send_query(':VOLD:HV?'))

    def offset(self, channel) -> float:
        self.select_channel(channel)
        return float(self.send_query(':VOLT:OFFS?'.format(channel=channel)))


TaborProgramMemory = NamedTuple('TaborProgramMemory', [('segment_indices', np.ndarray),
                                                       ])

class TaborChannelPair(AWG):
    def __init__(self, tabor_device: TaborAWGRepresentation, channels: Tuple[int, int], identifier: str):
        super().__init__(identifier)
        self.__device = tabor_device

        if channels not in ((1, 2), (3, 4)):
            raise ValueError('Invalid channel pair: {}'.format(channels))
        self.__channels = channels

        self.__segment_lengths = np.zeros(0, dtype=int)
        self.__segment_capacity = np.zeros(0, dtype=int)
        self.__segment_hashes = np.zeros(0, dtype=int)
        self.__segment_references = np.zeros(0, dtype=int)

        self.total_capacity = int(16e6)

        self.__known_programs = dict()  # type: Dict[str, TaborProgramMemory]

    def free_program(self, name: str) -> TaborProgramMemory:
        program = self.__known_programs.pop(name)
        self.__segment_references[program.segment_indices] -= 1
        return program

    @property
    def __segment_reserved(self) -> np.ndarray:
        return self.__segment_references > 0

    @property
    def __free_points_in_total(self) -> int:
        return self.total_capacity - np.sum(self.__segment_capacity[self.__segment_reserved])

    @property
    def __free_points_at_end(self) -> int:
        reserved_index = np.where(self.__segment_reserved)
        if reserved_index:
            return self.total_capacity - np.sum(self.__segment_capacity[:reserved_index[-1]])
        else:
            return self.total_capacity

    def upload(self, name: str,
               program: Loop,
               channels: List[ChannelID],
               markers: List[ChannelID],
               force: bool=False) -> None:
        """Upload a program to the AWG.

        The policy is to prefer amending the unknown waveforms to overwriting old ones."""

        if len(channels) != self.num_channels:
            raise ValueError('Channel ID not specified')
        if len(markers) != self.num_markers:
            raise ValueError('Markers not specified')

        # helper to restore previous state if upload is impossible
        to_restore = None
        if name in self.__known_programs:
            if force:
                # save old program to restore in on error
                to_restore = self.free_program(name)
            else:
                raise ValueError('{} is already known on {}'.format(name, self.identifier))

        try:
            # parse to tabor program
            tabor_program = TaborProgram(program, channels=tuple(channels))
            sample_rate = self.__device.sample_rate(self.__channels[0])
            voltage_amplitudes = (self.__device.amplitude(self.__channels[0]),
                                  self.__device.amplitude(self.__channels[1]))
            voltage_offsets = (self.__device.offset(self.__channels[0]),
                               self.__device.offset(self.__channels[1]))
            segments, segment_lengths = tabor_program.sampled_segments(sample_rate=sample_rate,
                                                      voltage_amplitude=voltage_amplitudes,
                                                      voltage_offset=voltage_offsets)
            segment_hashes = np.fromiter((hash(bytes(segment)) for segment in segments), count=len(segments), dtype=int)

            known_waveforms = np.in1d(segment_hashes, self.__segment_hashes, assume_unique=True)
            to_upload_size = np.sum(segment_lengths[~known_waveforms] + 16)

            waveform_to_segment = np.full(len(segments), -1, dtype=int)
            waveform_to_segment[known_waveforms] = np.where(
                np.in1d(self.__segment_hashes, segment_hashes[known_waveforms]))

            if name not in self.__known_programs:
                if self.__free_points_in_total < to_upload_size:
                    raise MemoryError('Not enough free memory')
                if self.__free_points_at_end < to_upload_size:
                    reserved_indices = np.where(self.__segment_reserved)
                    if len(reserved_indices) == 0:
                        raise MemoryError('Fragmentation does not allow upload.')

                    last_reserved = reserved_indices[-1] if reserved_indices else 0
                    free_segments = np.where(self.__segment_references[:last_reserved] == 0)[
                        np.argsort(self.__segment_capacity[:last_reserved])[::-1]]

                    to_amend = ~known_waveforms
                    to_insert = []
                    for wf_index in np.argsort(segment_lengths[~known_waveforms])[::-1]:
                        if segment_lengths[wf_index] <= self.__segment_capacity[free_segments[0]]:
                            to_insert.append((wf_index, free_segments[0]))
                            free_segments = free_segments[1:]
                            to_amend[wf_index] = False

                    if np.sum(segment_lengths[to_amend] + 16) > self.__free_points_at_end:
                        raise MemoryError('Fragmentation does not allow upload.')

        except:
            if to_restore:
                self.__known_programs[name] = to_restore
                self.__segment_reserved[to_restore.segment_indices] += 1
            raise

        self.__segment_references[waveform_to_segment[waveform_to_segment >= 0]] += 1

        if to_insert:
            # as we have to insert waveforms the waveforms behind the last referenced are discarded
            self.cleanup()

            for wf_index, segment_index in to_insert:
                self.__upload_segment(segment_index, segments[wf_index])
                waveform_to_segment[wf_index] = segment_index

        if np.any(to_amend):
            segments_to_amend = segments[to_amend]
            self.__amend_segments(segments_to_amend)
            waveform_to_segment[to_amend] = np.arange(len(self.__segment_capacity)-np.sum(to_amend),
                                                      len(self.__segment_capacity), dtype=int)

        self.__known_programs[name] = TaborProgramMemory(segment_index=waveform_to_segment,
                                                         )
        raise NotImplementedError()

    def __upload_segment(self, segment_index: int, segment: TaborSegment) -> None:
        self.__segment_references[segment_index] = 1
        self.__segment_hashes[segment_index]
        raise NotImplementedError()

    def __amend_segments(self, segments: List[TaborSegment]) -> None:
        raise NotImplementedError()

    def cleanup(self) -> None:
        """Discard all segments after the last which is still referenced"""
        reserved_indices = np.where(self.__segment_references > 0)

        new_end = reserved_indices[-1]+1 if reserved_indices else 0
        self.__segment_lengths = self.__segment_lengths[:new_end]
        self.__segment_capacity = self.__segment_capacity[:new_end]
        self.__segment_hashes = self.__segment_capacity[:new_end]
        self.__segment_references = self.__segment_capacity[:new_end]

    def remove(self, name: str) -> None:
        """Remove a program from the AWG.

        Also discards all waveforms referenced only by the program identified by name.

        Args:
            name (str): The name of the program to remove.
        """
        self.free_program(name)

    def arm(self, name: str) -> None:
        raise NotImplementedError()

    @property
    def programs(self) -> Set[str]:
        """The set of program names that can currently be executed on the hardware AWG."""
        raise set(program.name for program in self.__known_programs.keys())

    @property
    def sample_rate(self) -> float:
        return self.__device.sample_rate(self.__channels[0])

    @property
    def num_channels(self) -> int:
        return 2

    @property
    def num_markers(self) -> int:
        return 2


class TaborException(Exception):
    pass
