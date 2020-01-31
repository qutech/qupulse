import sys
from typing import NamedTuple, Optional, List, Generator, Tuple, Sequence, Mapping, Union, Dict, FrozenSet
from enum import Enum
import operator
from collections import OrderedDict
import itertools

import numpy as np

from qupulse.utils.types import ChannelID, TimeType
from qupulse.hardware.awgs.base import ProgramEntry
from qupulse.hardware.util import make_combined_wave, get_sample_times, voltage_to_uint16
from qupulse.pulses.parameters import Parameter, MappedParameter
from qupulse._program.waveforms import Waveform
from qupulse._program._loop import Loop

assert(sys.byteorder == 'little')


TableEntry = NamedTuple('TableEntry', [('repetition_count', int),
                                       ('element_number', int),
                                       ('jump_flag', int)])
TableEntry.__doc__ = ("Entry in sequencing or advanced sequencer table as uploaded to the AWG with"
                      "download_adv_seq_table or download_sequencer_table")

TableDescription = NamedTuple('TableDescription', [('repetition_count', int),
                                                   ('element_id', int),
                                                   ('jump_flag', int)])
TableDescription.__doc__ = ("Entry in sequencing or advanced sequencer table but with an element 'reference' instead of"
                            "the hardware bound 'number'")

PositionalEntry = NamedTuple('PositionalEntry', [('position', int),
                                                 ('element_number', int),
                                                 ('repetition_count', int),
                                                 ('jump_flag', int)])
PositionalEntry.__doc__ = ("Entry in sequencing or advanced sequencer table as uploaded with :SEQ:DEF"
                           "or :ASEQ:DEF")


class TaborException(Exception):
    pass


class TaborSegment:
    """Represents one segment of two channels on the device. Convenience class."""

    __slots__ = ('ch_a', 'ch_b', 'marker_a', 'marker_b')

    def __init__(self,
                 ch_a: Optional[np.ndarray],
                 ch_b: Optional[np.ndarray],
                 marker_a: Optional[np.ndarray],
                 marker_b: Optional[np.ndarray]):
        if ch_a is None and ch_b is None:
            raise TaborException('Empty TaborSegments are not allowed')
        if ch_a is not None and ch_b is not None and len(ch_a) != len(ch_b):
            raise TaborException('Channel entries have to have the same length')

        self.ch_a = None if ch_a is None else np.asarray(ch_a, dtype=np.uint16)
        self.ch_b = None if ch_b is None else np.asarray(ch_b, dtype=np.uint16)

        self.marker_a = None if marker_a is None else np.asarray(marker_a, dtype=bool)
        self.marker_b = None if marker_b is None else np.asarray(marker_b, dtype=bool)

        if marker_a is not None and len(marker_a)*2 != self.num_points:
            raise TaborException('Marker A has to have half of the channels length')
        if marker_b is not None and len(marker_b)*2 != self.num_points:
            raise TaborException('Marker B has to have half of the channels length')

    @classmethod
    def from_binary_segment(cls, segment_data: np.ndarray) -> 'TaborSegment':
        data_a = segment_data.reshape((-1, 16))[1::2, :].reshape((-1, ))
        data_b = segment_data.reshape((-1, 16))[0::2, :].ravel()
        return cls.from_binary_data(data_a, data_b)

    @classmethod
    def from_binary_data(cls, data_a: np.ndarray, data_b: np.ndarray) -> 'TaborSegment':
        ch_b = data_b

        channel_mask = np.uint16(2**14 - 1)
        ch_a = np.bitwise_and(data_a, channel_mask)

        marker_a_mask = np.uint16(2**14)
        marker_b_mask = np.uint16(2**15)
        marker_data = data_a.reshape(-1, 8)[1::2, :].reshape((-1, ))

        marker_a = np.bitwise_and(marker_data, marker_a_mask)
        marker_b = np.bitwise_and(marker_data, marker_b_mask)

        return cls(ch_a=ch_a,
                   ch_b=ch_b,
                   marker_a=marker_a,
                   marker_b=marker_b)

    def __hash__(self) -> int:
        return hash(tuple(0 if data is None else bytes(data)
                          for data in (self.ch_a, self.ch_b, self.marker_a, self.marker_b)))

    def __eq__(self, other: 'TaborSegment'):
        def compare_markers(marker_1, marker_2):
            if marker_1 is None:
                if marker_2 is None:
                    return True
                else:
                    return not np.any(marker_2)

            elif marker_2 is None:
                return not np.any(marker_1)

            else:
                return np.array_equal(marker_1, marker_2)

        return (np.array_equal(self.ch_a, other.ch_a) and
                np.array_equal(self.ch_b, other.ch_b) and
                compare_markers(self.marker_a, other.marker_a) and
                compare_markers(self.marker_b, other.marker_b))

    @property
    def data_a(self) -> np.ndarray:
        """channel_data and marker data"""
        if self.marker_a is None and self.marker_b is None:
            return self.ch_a

        if self.ch_a is None:
            raise NotImplementedError('What data should be used in a?')

        # copy channel information
        data = np.array(self.ch_a)

        if self.marker_a is not None:
            data.reshape(-1, 8)[1::2, :].flat |= (1 << 14) * self.marker_a.astype(np.uint16)

        if self.marker_b is not None:
            data.reshape(-1, 8)[1::2, :].flat |= (1 << 15) * self.marker_b.astype(np.uint16)

        return data

    @property
    def data_b(self) -> np.ndarray:
        """channel_data and marker data"""
        return self.ch_b

    @property
    def num_points(self) -> int:
        return len(self.ch_b) if self.ch_a is None else len(self.ch_a)

    def get_as_binary(self) -> np.ndarray:
        assert not (self.ch_a is None or self.ch_b is None)
        return make_combined_wave([self])


class PlottableProgram:
    def __init__(self,
                 segments: List[TaborSegment],
                 sequence_tables: List[List[Tuple[int, int, int]]],
                 advanced_sequence_table: List[Tuple[int, int, int]]):
        self._segments = segments
        self._sequence_tables = [[self.TableEntry(*sequence_table_entry)
                                  for sequence_table_entry in sequence_table]
                                 for sequence_table in sequence_tables]
        self._advanced_sequence_table = [self.TableEntry(*adv_seq_entry)
                                         for adv_seq_entry in advanced_sequence_table]

    @classmethod
    def from_read_data(cls, waveforms: List[np.ndarray],
                       sequence_tables: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                       advanced_sequence_table: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> 'PlottableProgram':
        return cls([TaborSegment.from_binary_segment(wf) for wf in waveforms],
                   [cls._reformat_rep_seg_jump(seq_table) for seq_table in sequence_tables],
                   cls._reformat_rep_seg_jump(advanced_sequence_table))

    @classmethod
    def _reformat_rep_seg_jump(cls, rep_seg_jump_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> List[TableEntry]:
        return list(cls.TableEntry(int(rep), int(seg_no), int(jump))
                    for rep, seg_no, jump in zip(*rep_seg_jump_tuple))

    def _get_advanced_sequence_table(self, with_first_idle=False, with_last_idles=False) -> List[TableEntry]:
        if not with_first_idle and self._advanced_sequence_table[0] == (1, 1, 1):
            adv_seq_tab = self._advanced_sequence_table[1:]
        else:
            adv_seq_tab = self._advanced_sequence_table

        #  remove idle pulse at end
        if with_last_idles:
            return adv_seq_tab
        else:
            while adv_seq_tab[-1] == (1, 1, 0):
                adv_seq_tab = adv_seq_tab[:-1]
            return adv_seq_tab

    def _iter_segment_table_entry(self,
                                  with_first_idle=False,
                                  with_last_idles=False) -> Generator[TableEntry, None, None]:
        for sequence_repeat, sequence_no, _ in self._get_advanced_sequence_table(with_first_idle, with_last_idles):
            for _ in range(sequence_repeat):
                yield from self._sequence_tables[sequence_no - 1]

    def iter_waveforms_and_repetitions(self,
                                       channel: int,
                                       with_first_idle=False,
                                       with_last_idles=False) -> Generator[Tuple[np.ndarray, int], None, None]:
        ch_getter = (operator.attrgetter('ch_a'), operator.attrgetter('ch_b'))[channel]
        for segment_repeat, segment_no, _ in self._iter_segment_table_entry(with_first_idle, with_last_idles):
            yield ch_getter(self._segments[segment_no - 1]), segment_repeat

    def iter_samples(self, channel: int,
                     with_first_idle=False,
                     with_last_idles=False) -> Generator[np.uint16, None, None]:
        for waveform, repetition in self.iter_waveforms_and_repetitions(channel, with_first_idle, with_last_idles):
            waveform = list(waveform)
            for _ in range(repetition):
                yield from waveform

    def get_as_single_waveform(self, channel: int,
                               max_total_length: int=10**9,
                               with_marker: bool=False) -> Optional[np.ndarray]:
        waveforms = self.get_waveforms(channel, with_marker=with_marker)
        repetitions = self.get_repetitions()
        waveform_lengths = np.fromiter((wf.size for wf in waveforms), count=len(waveforms), dtype=np.uint64)

        total_length = (repetitions*waveform_lengths).sum()
        if total_length > max_total_length:
            return None

        result = np.empty(total_length, dtype=np.uint16)
        c_idx = 0
        for wf, rep in zip(waveforms, repetitions):
            mem = wf.size*rep
            target = result[c_idx:c_idx+mem]

            target = target.reshape((rep, wf.size))
            target[:, :] = wf[np.newaxis, :]
            c_idx += mem
        return result

    def get_waveforms(self, channel: int, with_marker: bool=False) -> List[np.ndarray]:
        if with_marker:
            ch_getter = (operator.attrgetter('data_a'), operator.attrgetter('data_b'))[channel]
        else:
            ch_getter = (operator.attrgetter('ch_a'), operator.attrgetter('ch_b'))[channel]
        return [ch_getter(self._segments[segment_no - 1])
                for _, segment_no, _ in self._iter_segment_table_entry()]

    def get_segment_waveform(self, channel: int, segment_no: int) -> np.ndarray:
        ch_getter = (operator.attrgetter('ch_a'), operator.attrgetter('ch_b'))[channel]
        return [ch_getter(self._segments[segment_no - 1])]

    def get_repetitions(self) -> np.ndarray:
        return np.fromiter((segment_repeat
                            for segment_repeat, *_ in self._iter_segment_table_entry()), dtype=np.uint32)

    def __eq__(self, other):
        for ch in (0, 1):
            for x, y in itertools.zip_longest(self.iter_samples(ch, True, False),
                                              other.iter_samples(ch, True, False)):
                if x != y:
                    return False
        return True

    def to_builtin(self) -> dict:
        waveforms = [[wf.data_a.tolist() for wf in self._segments],
                     [wf.data_b.tolist() for wf in self._segments]]
        return {'waveforms': waveforms,
                'seq_tables': self._sequence_tables,
                'adv_seq_table': self._advanced_sequence_table}

    @classmethod
    def from_builtin(cls, data: dict) -> 'PlottableProgram':
        waveforms = data['waveforms']
        waveforms = [TaborSegment.from_binary_data(np.array(data_a, dtype=np.uint16), np.array(data_b, dtype=np.uint16))
                     for data_a, data_b in zip(*waveforms)]
        return cls(waveforms, data['seq_tables'], data['adv_seq_table'])


class TaborSequencing(Enum):
    SINGLE = 1
    ADVANCED = 2


class TaborProgram(ProgramEntry):
    """

    Implementations notes concerning indices / position
     - index: zero based index in internal data structure f.i. the waveform list
     - position: ?
     - no/number: one based index on device


    """

    def __init__(self,
                 program: Loop,
                 device_properties: Mapping,
                 channels: Tuple[Optional[ChannelID], Optional[ChannelID]],
                 markers: Tuple[Optional[ChannelID], Optional[ChannelID]],
                 amplitudes: Tuple[float, float],
                 offsets: Tuple[float, float],
                 voltage_transformations: Tuple[Optional[callable], Optional[callable]],
                 sample_rate: TimeType,
                 mode: TaborSequencing = None
                 ):
        if len(channels) != device_properties['chan_per_part']:
            raise TaborException('TaborProgram only supports {} channels'.format(device_properties['chan_per_part']))
        if len(markers) != device_properties['chan_per_part']:
            raise TaborException('TaborProgram only supports {} markers'.format(device_properties['chan_per_part']))
        used_channels = frozenset(set(channels).union(markers) - {None})

        if program.repetition_count > 1 or program.depth() == 0:
            program.encapsulate()

        if mode is None:
            if program.depth() > 1:
                mode = TaborSequencing.ADVANCED
            else:
                mode = TaborSequencing.SINGLE

        super().__init__(loop=program,
                         channels=channels,
                         markers=markers,
                         amplitudes=amplitudes,
                         offsets=offsets,
                         voltage_transformations=voltage_transformations,
                         sample_rate=sample_rate,
                         waveforms=None  # no sampling happens here
                         )

        self._used_channels = used_channels
        self._parsed_program = None  # type: Optional[ParsedProgram]
        self._mode = None
        self._device_properties = device_properties

        assert mode in (TaborSequencing.ADVANCED, TaborSequencing.SINGLE), "Invalid mode"
        if mode == TaborSequencing.SINGLE:
            self.setup_single_sequence_mode()
        else:
            self.setup_advanced_sequence_mode()

        self._sampled_segments = self._calc_sampled_segments()

    @property
    def markers(self) -> Tuple[Optional[ChannelID], Optional[ChannelID]]:
        return self._markers

    @property
    def channels(self) -> Tuple[Optional[ChannelID], Optional[ChannelID]]:
        return self._channels

    @property
    def program(self) -> Loop:
        return self._loop

    def _channel_data(self, waveform: Waveform, time: np.ndarray, channel_idx: int):
        if self._channels[channel_idx] is None:
            return np.full_like(time, 8192, dtype=np.uint16)

        else:
            return voltage_to_uint16(
                self._voltage_transformations[channel_idx](
                    waveform.get_sampled(channel=self._channels[channel_idx],
                                         sample_times=time)),
                self._amplitudes[channel_idx],
                self._offsets[channel_idx],
                resolution=14)

    def _marker_data(self, waveform: Waveform, time: np.ndarray, marker_idx: int):
        if self._markers[marker_idx] is None:
            return np.full_like(time, False, dtype=bool)
        else:
            marker = self._markers[marker_idx]
            return waveform.get_sampled(channel=marker, sample_times=time) != 0

    def _calc_sampled_segments(self) -> Tuple[Sequence[TaborSegment], Sequence[int]]:
        """
        Returns:
            (segments, segment_lengths)
        """
        time_array, segment_lengths = get_sample_times(self._waveforms, self._sample_rate)

        if np.any(segment_lengths % 16 > 0) or np.any(segment_lengths < 192):
            raise TaborException('At least one waveform has a length that is smaller 192 or not a multiple of 16')

        segments = []
        for i, waveform in enumerate(self._waveforms):
            t = time_array[:segment_lengths[i]]
            marker_time = t[::2]
            segment_a = self._channel_data(waveform, t, 0)
            segment_b = self._channel_data(waveform, t, 1)
            assert (len(segment_a) == len(t))
            assert (len(segment_b) == len(t))
            marker_a = self._marker_data(waveform, marker_time, 0)
            marker_b = self._marker_data(waveform, marker_time, 1)
            segment = TaborSegment(ch_a=segment_a,
                                   ch_b=segment_b,
                                   marker_a=marker_a,
                                   marker_b=marker_b)
            segments.append(segment)
        return segments, segment_lengths

    def setup_single_sequence_mode(self) -> None:
        assert self.program.depth() == 1
        assert self.program.is_balanced()
        self._parsed_program = parse_single_seq_program(program=self.program, used_channels=self._used_channels)
        self._mode = TaborSequencing.SINGLE

    def setup_advanced_sequence_mode(self) -> None:
        assert self.program.depth() > 1
        assert self.program.repetition_count == 1

        self.program.flatten_and_balance(2)

        min_seq_len = self._device_properties['min_seq_len']
        max_seq_len = self._device_properties['max_seq_len']

        for sequence_table in self.program:
            assert len(sequence_table) >= min_seq_len
            assert len(sequence_table) <= max_seq_len

        self._parsed_program = parse_aseq_program(self.program, used_channels=self._used_channels)
        self._mode = TaborSequencing.ADVANCED

    def get_sampled_segments(self) -> Tuple[Sequence[TaborSegment], Sequence[int]]:
        return self._sampled_segments

    def update_volatile_parameters(self, parameters: Mapping[str, Parameter]) -> Mapping[Union[int, Tuple[int, int]],
                                                                                         Union[TableEntry, TableDescription]]:
        """

        Args:
            parameters:

        Returns:
            Mapping position of change -> (new repetition value, element_num/id, jump flag)
        """
        modifications = {}

        for position, parameter in self._parsed_program.volatile_parameter_positions.items():
            parameter.update_constants(parameters)
            new_value = int(parameter.get_value())

            if isinstance(position, int):
                old_rep_count, element_num, jump_flag = self._parsed_program.advanced_sequencer_table[position]
                self._parsed_program.advanced_sequencer_table[position] = TableEntry(new_value, element_num, jump_flag)
                if new_value != old_rep_count:
                    modifications[position] = TableEntry(repetition_count=new_value,
                                                         element_number=element_num, jump_flag=jump_flag)
            else:
                adv_pos, seq_pos = position
                ((old_rep_count, element_id, jump_flag), param) = self._parsed_program.sequencer_tables[adv_pos][seq_pos]

                self._parsed_program.sequencer_tables[adv_pos][seq_pos] = (TableDescription(new_value, element_id,
                                                                                            jump_flag), param)
                if new_value != old_rep_count:
                    modifications[position] = TableDescription(repetition_count=new_value,
                                                               element_id=element_id, jump_flag=jump_flag)

        return modifications

    def get_sequencer_tables(self):  # -> List[List[TableDescription, Optional[MappedParameter]]]:
        return self._parsed_program.sequencer_tables

    def get_advanced_sequencer_table(self) -> List[TableEntry]:
        """Advanced sequencer table that can be used  via the download_adv_seq_table pytabor command"""
        return self._parsed_program.advanced_sequencer_table

    @property
    def waveform_mode(self) -> str:
        return self._mode


def _check_merge_with_next(program, n, max_seq_len):
    if (program[n].repetition_count == 1 and program[n+1].repetition_count == 1 and
            len(program[n]) + len(program[n+1]) < max_seq_len):
        program[n][len(program[n]):] = program[n + 1][:]
        program[n + 1:n + 2] = []
        return True
    return False


def _check_partial_unroll(program, n, min_seq_len):
    st = program[n]
    if st.repetition_parameter is not None:
        return False

    if sum(entry.repetition_count for entry in st) * st.repetition_count >= min_seq_len:
        if sum(entry.repetition_count for entry in st) < min_seq_len:
            st.unroll_children()
        while len(st) < min_seq_len:
            st.split_one_child()
        return True
    return False


def prepare_program_for_advanced_sequence_mode(program: Loop, min_seq_len, max_seq_len):
    i = 0
    while i < len(program):
        program[i].assert_tree_integrity()
        if len(program[i]) > max_seq_len:
            raise TaborException('The algorithm is not smart enough to make sequence tables shorter')
        elif len(program[i]) < min_seq_len:
            assert program[i].repetition_count > 0
            if program[i].repetition_count == 1:
                # check if merging with neighbour is possible
                if i > 0 and _check_merge_with_next(program, i - 1, max_seq_len=max_seq_len):
                    pass
                elif i + 1 < len(program) and _check_merge_with_next(program, i, max_seq_len=max_seq_len):
                    pass

                # check if (partial) unrolling is possible
                elif _check_partial_unroll(program, i, min_seq_len=min_seq_len):
                    i += 1

                # check if sequence table can be extended by unrolling a neighbor
                elif (i > 0
                      and program[i - 1].repetition_count > 1
                      and len(program[i]) + len(program[i - 1]) < max_seq_len):
                    program[i][:0] = program[i - 1].copy_tree_structure()[:]
                    program[i - 1].repetition_count -= 1

                elif (i + 1 < len(program)
                      and program[i + 1].repetition_count > 1
                      and len(program[i]) + len(program[i + 1]) < max_seq_len):
                    program[i][len(program[i]):] = program[i + 1].copy_tree_structure()[:]
                    program[i + 1].repetition_count -= 1

                else:
                    raise TaborException('The algorithm is not smart enough to make this sequence table longer')
            elif _check_partial_unroll(program, i, min_seq_len=min_seq_len):
                i += 1
            else:
                raise TaborException('The algorithm is not smart enough to make this sequence table longer')
        else:
            i += 1


ParsedProgram = NamedTuple('ParsedProgram', [('advanced_sequencer_table', List[TableEntry]),
                                                 ('sequencer_tables', List[List[Tuple[TableDescription,
                                                                                      Optional[MappedParameter]]]]),
                                                 ('waveforms', Tuple[Waveform, ...]),
                                                 ('volatile_parameter_positions', Dict[Union[int, Tuple[int, int]],
                                                                                       MappedParameter])])


def parse_aseq_program(program: Loop, used_channels: FrozenSet[ChannelID]) -> ParsedProgram:
    volatile_parameter_positions = {}

    advanced_sequencer_table = []
    sequencer_tables = []
    waveforms = OrderedDict()
    for adv_position, sequencer_table_loop in enumerate(program):
        current_sequencer_table = []
        for position, (waveform, repetition_count, repetition_parameter) in enumerate(
                (waveform_loop.waveform.get_subset_for_channels(used_channels),
                 waveform_loop.repetition_count, waveform_loop.repetition_parameter)
                for waveform_loop in sequencer_table_loop):
            if waveform in waveforms:
                wf_index = waveforms[waveform]
            else:
                wf_index = len(waveforms)
                waveforms[waveform] = wf_index

            # TODO: use hashable parameter representation and use an ordered dict
            current_sequencer_table.append((TableDescription(repetition_count=repetition_count,
                                                             element_id=wf_index, jump_flag=0),
                                            repetition_parameter))

            if repetition_parameter is not None:
                volatile_parameter_positions[(adv_position, position)] = repetition_parameter

        if current_sequencer_table in sequencer_tables:
            sequence_no = sequencer_tables.index(current_sequencer_table) + 1

        else:
            sequence_no = len(sequencer_tables) + 1
            sequencer_tables.append(current_sequencer_table)

        advanced_sequencer_table.append(TableEntry(sequencer_table_loop.repetition_count, sequence_no, 0))
        if sequencer_table_loop.repetition_parameter is not None:
            volatile_parameter_positions[adv_position] = sequencer_table_loop.repetition_parameter

    return ParsedProgram(
        advanced_sequencer_table=advanced_sequencer_table,
        sequencer_tables=sequencer_tables,
        waveforms=tuple(waveforms.keys()),
        volatile_parameter_positions=volatile_parameter_positions
    )


def parse_single_seq_program(program: Loop, used_channels: FrozenSet[ChannelID]) -> ParsedProgram:
    assert program.depth() == 1

    sequencer_table = []
    waveforms = OrderedDict()
    volatile_parameter_positions = {}

    for position, (waveform, repetition_count, repetition_parameter) in enumerate(
            (waveform_loop.waveform.get_subset_for_channels(used_channels),
             waveform_loop.repetition_count, waveform_loop.repetition_parameter)
            for waveform_loop in program):
        if waveform in waveforms:
            waveform_index = waveforms[waveform]
        else:
            waveform_index = len(waveforms)
            waveforms[waveform] = waveform_index

        sequencer_table.append((TableDescription(repetition_count=repetition_count,
                                                 element_id=waveform_index,
                                                 jump_flag=0), repetition_parameter))
        if repetition_parameter is not None:
            volatile_parameter_positions[(0, position)] = repetition_parameter

    return ParsedProgram(
        advanced_sequencer_table=[TableEntry(repetition_count=program.repetition_count, element_number=1, jump_flag=0)],
        sequencer_tables=[sequencer_table],
        waveforms=tuple(waveforms.keys()),
        volatile_parameter_positions=volatile_parameter_positions
    )
