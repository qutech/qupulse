import sys
from typing import NamedTuple, Optional, List, Generator, Tuple, Sequence, Mapping, Union, Dict, FrozenSet, cast
from enum import Enum
import operator
from collections import OrderedDict
import itertools
import numbers

import numpy as np

from qupulse.utils.types import ChannelID, TimeType
from qupulse.hardware.awgs.base import ProgramEntry
from qupulse.hardware.util import get_sample_times, voltage_to_uint16
from qupulse.pulses.parameters import Parameter
from qupulse._program.waveforms import Waveform
from qupulse._program._loop import Loop
from qupulse._program.volatile import VolatileRepetitionCount, VolatileProperty

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
    """Represents one segment of two channels on the device. Convenience class. The data is stored in native format"""

    QUANTUM = 16
    ZERO_VAL = np.uint16(8192)
    CHANNEL_MASK = np.uint16(2**14 - 1)
    MARKER_A_MASK = np.uint16(2**14)
    MARKER_B_MASK = np.uint16(2**15)

    __slots__ = ('_data', '_hash')

    @staticmethod
    def _data_a_view(data: np.ndarray, writable: bool) -> np.ndarray:
        view = data[:, 1, :]
        assert not writable or view.base is data
        return view

    @staticmethod
    def _data_b_view(data: np.ndarray, writable: bool) -> np.ndarray:
        view = data[:, 0, :]
        assert not writable or view.base is data
        return view

    @staticmethod
    def _marker_data_view(data: np.ndarray, writable: bool) -> np.ndarray:
        view = data.reshape((-1, 2, 2, 8))[:, 1, 1, :]
        assert not writable or view.base is data
        return view

    def __init__(self, *,
                 data: np.ndarray):
        assert data.ndim == 3
        n_quanta, n_channels, quantum_size = data.shape
        assert n_channels == 2
        assert quantum_size == self.QUANTUM
        assert data.dtype is np.dtype('uint16')
        self._data = data
        self._data.flags.writeable = False

        # shape is not included because it only depends on the size i.e. (n_quantum, 2, 16)
        self._hash = hash(self._data.tobytes())

    @property
    def native(self) -> np.ndarray:
        """You must not change native (data or shape)

        Returns:
            An array with shape (n_quanta, 2, 16)
        """
        return self._data

    @property
    def data_a(self) -> np.ndarray:
        return self._data_a_view(self._data, writable=False).reshape(-1)

    @property
    def data_b(self) -> np.ndarray:
        return self._data_b_view(self._data, writable=False).reshape(-1)

    @classmethod
    def from_sampled(cls,
                     ch_a: Optional[np.ndarray],
                     ch_b: Optional[np.ndarray],
                     marker_a: Optional[np.ndarray],
                     marker_b: Optional[np.ndarray]) -> 'TaborSegment':
        num_points = set()
        if ch_a is not None:
            assert ch_a.ndim == 1
            assert ch_a.dtype is np.dtype('uint16')
            num_points.add(ch_a.size)
        if ch_b is not None:
            assert ch_b.ndim == 1
            assert ch_b.dtype is np.dtype('uint16')
            num_points.add(ch_b.size)
        if marker_a is not None:
            assert marker_a.ndim == 1
            marker_a = marker_a.astype(dtype=bool)
            num_points.add(marker_a.size * 2)
        if marker_b is not None:
            assert marker_b.ndim == 1
            marker_b = marker_b.astype(dtype=bool)
            num_points.add(marker_b.size * 2)

        if len(num_points) == 0:
            raise TaborException('Empty TaborSegments are not allowed')
        elif len(num_points) > 1:
            raise TaborException('Channel entries have to have the same length')
        num_points, = num_points

        assert num_points % cls.QUANTUM == 0

        data = np.full((num_points // cls.QUANTUM, 2, cls.QUANTUM), cls.ZERO_VAL, dtype=np.uint16)
        data_a = cls._data_a_view(data, writable=True)
        data_b = cls._data_b_view(data, writable=True)
        marker_view = cls._marker_data_view(data, writable=True)

        if ch_a is not None:
            data_a[:] = ch_a.reshape((-1, cls.QUANTUM))

        if ch_b is not None:
            data_b[:] = ch_b.reshape((-1, cls.QUANTUM))

        if marker_a is not None:
            marker_view[:] |= np.left_shift(marker_a.astype(np.uint16), 14).reshape((-1, 8))

        if marker_b is not None:
            marker_view[:] |= np.left_shift(marker_b.astype(np.uint16), 15).reshape((-1, 8))

        return cls(data=data)

    @classmethod
    def from_binary_segment(cls, segment_data: np.ndarray) -> 'TaborSegment':
        return cls(data=segment_data.reshape((-1, 2, 16)))

    @property
    def ch_a(self):
        return np.bitwise_and(self.data_a, self.CHANNEL_MASK)

    @property
    def ch_b(self):
        return self.data_b

    @property
    def marker_a(self) -> np.ndarray:
        marker_data = self._marker_data_view(self._data, writable=False)
        return np.bitwise_and(marker_data, self.MARKER_A_MASK).astype(bool).reshape(-1)

    @property
    def marker_b(self) -> np.ndarray:
        marker_data = self._marker_data_view(self._data, writable=False)
        return np.bitwise_and(marker_data, self.MARKER_B_MASK).astype(bool).reshape(-1)

    @classmethod
    def from_binary_data(cls, data_a: np.ndarray, data_b: np.ndarray) -> 'TaborSegment':
        assert data_a.size == data_b.size
        assert data_a.ndim == 1 == data_b.ndim
        assert data_a.size % 16 == 0

        data = np.empty((data_a.size // 16, 2, 16), dtype=np.uint16)
        cls._data_a_view(data, writable=True)[:] = data_a.reshape((-1, 16))
        cls._data_b_view(data, writable=True)[:] = data_b.reshape((-1, 16))

        return cls(data=data)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: 'TaborSegment'):
        return np.array_equal(self._data, other._data)

    @property
    def num_points(self) -> int:
        return self._data.shape[0] * 16

    def get_as_binary(self) -> np.ndarray:
        return self.native.ravel()


def make_combined_wave(segments: List[TaborSegment], destination_array=None) -> np.ndarray:
    """Combine multiple segments to one binary blob for bulk upload. Better implementation of
    `pytabor.make_combined_wave`.

    Args:
        segments:
        destination_array:

    Returns:
        1 d array for upload to instrument
    """
    quantum = TaborSegment.QUANTUM

    if len(segments) == 0:
        return np.zeros(0, dtype=np.uint16)

    n_quanta = sum(segment.native.shape[0] for segment in segments) + len(segments) - 1

    if destination_array is not None:
        if destination_array.size != 2 * n_quanta * quantum:
            raise ValueError('Destination array has an invalid size')
        destination_array = destination_array.reshape((n_quanta, 2, quantum))
    else:
        destination_array = np.empty((n_quanta, 2, quantum), dtype=np.uint16)

    current_quantum = 0
    for segment in segments:
        if current_quantum > 0:
            # fill one quantum with first data point from upcoming segment
            destination_array[current_quantum, :, :] = segment.native[0, :, 0][:, None]
            current_quantum += 1

        segment_quanta = segment.native.shape[0]
        destination_array[current_quantum:current_quantum + segment_quanta, ...] = segment.native

        current_quantum += segment_quanta

    return destination_array.ravel()


class PlottableProgram:
    def __init__(self,
                 segments: List[TaborSegment],
                 sequence_tables: List[List[Tuple[int, int, int]]],
                 advanced_sequence_table: List[Tuple[int, int, int]]):
        self._segments = segments
        self._sequence_tables = [[TableEntry(*sequence_table_entry)
                                  for sequence_table_entry in sequence_table]
                                 for sequence_table in sequence_tables]
        self._advanced_sequence_table = [TableEntry(*adv_seq_entry)
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
        return list(TableEntry(int(rep), int(seg_no), int(jump))
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
                 mode: TaborSequencing = None,
                 repetition_mode: str = "infinite",
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
                         waveforms=[]  # no sampling happens here
                         )

        self._used_channels = used_channels
        self._parsed_program = None  # type: Optional[ParsedProgram]
        self._mode = None
        self._device_properties = device_properties
        self._repetition_mode = repetition_mode

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
        time_array, segment_lengths = get_sample_times(self._parsed_program.waveforms, self._sample_rate)

        if np.any(segment_lengths % 16 > 0) or np.any(segment_lengths < 192):
            raise TaborException('At least one waveform has a length that is smaller 192 or not a multiple of 16')

        segments = []
        for i, waveform in enumerate(self._parsed_program.waveforms):
            t = time_array[:segment_lengths[i]]
            marker_time = t[::2]
            segment_a = self._channel_data(waveform, t, 0)
            segment_b = self._channel_data(waveform, t, 1)
            assert (len(segment_a) == len(t))
            assert (len(segment_b) == len(t))
            marker_a = self._marker_data(waveform, marker_time, 0)
            marker_b = self._marker_data(waveform, marker_time, 1)
            segment = TaborSegment.from_sampled(ch_a=segment_a,
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

        prepare_program_for_advanced_sequence_mode(self.program, min_seq_len=min_seq_len, max_seq_len=max_seq_len)

        for sequence_table in self.program:
            assert len(sequence_table) >= min_seq_len
            assert len(sequence_table) <= max_seq_len

        self._parsed_program = parse_aseq_program(self.program, used_channels=self._used_channels)
        self._mode = TaborSequencing.ADVANCED

    def get_sampled_segments(self) -> Tuple[Sequence[TaborSegment], Sequence[int]]:
        return self._sampled_segments

    def update_volatile_parameters(self, parameters: Mapping[str, numbers.Number]) -> Mapping[Union[int, Tuple[int, int]],
                                                                                         Union[TableEntry, TableDescription]]:
        """ Set the values of parameters which were marked as volatile on program creation. Sets volatile parameters
        in program memory.

        Args:
            parameters: Name of volatile parameters and respective values to which they should be set.

        Returns:
            Mapping position of change -> (new repetition value, element_num/id, jump flag)
        """
        modifications = {}

        for position, volatile_repetition in self._parsed_program.volatile_parameter_positions.items():
            if isinstance(position, int):
                old_rep_count, element_num, jump_flag = self._parsed_program.advanced_sequencer_table[position]
                new_value = volatile_repetition.update_volatile_dependencies(parameters)

                if new_value != old_rep_count:
                    new_entry = TableEntry(repetition_count=new_value, element_number=element_num, jump_flag=jump_flag)
                    self._parsed_program.advanced_sequencer_table[position] = new_entry
                    modifications[position] = new_entry
            else:
                adv_idx, seq_pos = position
                adv_pos = self._parsed_program.advanced_sequencer_table[adv_idx].element_number - 1
                sequencer_table = self._parsed_program.sequencer_tables[adv_pos]
                ((old_rep_count, element_id, jump_flag), param) = sequencer_table[seq_pos]

                new_value = volatile_repetition.update_volatile_dependencies(parameters)
                if new_value != old_rep_count:
                    new_description = TableDescription(repetition_count=new_value,
                                                       element_id=element_id, jump_flag=jump_flag)
                    sequencer_table[seq_pos] = (new_description, param)
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
    if st.volatile_repetition:
        return False

    if sum(entry.repetition_count for entry in st) * st.repetition_count >= min_seq_len:
        if sum(entry.repetition_count for entry in st) < min_seq_len:
            st.unroll_children()
        while len(st) < min_seq_len:
            st.split_one_child()
        return True
    return False


def prepare_program_for_advanced_sequence_mode(program: Loop, min_seq_len: int, max_seq_len: int):
    """This function tries to bring the program in a form, where the sequence tables' lengths are valid.

    Args:
        program:
        min_seq_len:
        max_seq_len:

    Raises:
        TaborException: if a sequence table that is too long cannot be shortened or a sequence table that is to short
            cannot be enlarged.
    """
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
                    if program[i + 1].volatile_repetition:
                        program[i + 1].volatile_repetition = program[i + 1].volatile_repetition - 1

                else:
                    raise TaborException('The algorithm is not smart enough to make this sequence table longer')
            elif _check_partial_unroll(program, i, min_seq_len=min_seq_len):
                i += 1
            else:
                raise TaborException('The algorithm is not smart enough to make this sequence table longer')
        else:
            i += 1


ParsedProgram = NamedTuple('ParsedProgram', [('advanced_sequencer_table', Sequence[TableEntry]),
                                             ('sequencer_tables', Sequence[Sequence[
                                                     Tuple[TableDescription, Optional[VolatileProperty]]]]),
                                             ('waveforms', Tuple[Waveform, ...]),
                                             ('volatile_parameter_positions', Dict[Union[int, Tuple[int, int]],
                                                                                   VolatileRepetitionCount])])


def parse_aseq_program(program: Loop, used_channels: FrozenSet[ChannelID]) -> ParsedProgram:
    volatile_parameter_positions = {}

    advanced_sequencer_table = []
    # we use an ordered dict here to avoid O(n**2) behaviour while looking for duplicates
    sequencer_tables = OrderedDict()
    waveforms = OrderedDict()
    for adv_position, sequencer_table_loop in enumerate(program):
        current_sequencer_table = []
        for position, (waveform, repetition_definition, volatile_repetition) in enumerate(
                (waveform_loop.waveform.get_subset_for_channels(used_channels),
                 waveform_loop.repetition_definition, waveform_loop.volatile_repetition)
                for waveform_loop in cast(Sequence[Loop], sequencer_table_loop)):

            wf_index = waveforms.setdefault(waveform, len(waveforms))
            current_sequencer_table.append((TableDescription(repetition_count=int(repetition_definition),
                                                             element_id=wf_index, jump_flag=0),
                                            volatile_repetition))

            if volatile_repetition:
                assert not isinstance(repetition_definition, int)
                volatile_parameter_positions[(adv_position, position)] = repetition_definition

        # make hashable
        current_sequencer_table = tuple(current_sequencer_table)

        sequence_index = sequencer_tables.setdefault(current_sequencer_table, len(sequencer_tables))
        sequence_no = sequence_index + 1

        advanced_sequencer_table.append(TableEntry(repetition_count=sequencer_table_loop.repetition_count,
                                                   element_number=sequence_no, jump_flag=0))
        if sequencer_table_loop.volatile_repetition:
            volatile_parameter_positions[adv_position] = sequencer_table_loop.repetition_definition

    # transform sequencer_tables in lists to make it indexable and mutable
    sequencer_tables = list(map(list, sequencer_tables))

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

    for position, (waveform, repetition_definition, volatile_repetition) in enumerate(
            (waveform_loop.waveform.get_subset_for_channels(used_channels),
             waveform_loop.repetition_definition, waveform_loop.volatile_repetition)
            for waveform_loop in program):
        if waveform in waveforms:
            waveform_index = waveforms[waveform]
        else:
            waveform_index = len(waveforms)
            waveforms[waveform] = waveform_index

        sequencer_table.append((TableDescription(repetition_count=int(repetition_definition),
                                                 element_id=waveform_index,
                                                 jump_flag=0), volatile_repetition))
        if volatile_repetition is not None:
            volatile_parameter_positions[(0, position)] = repetition_definition

    return ParsedProgram(
        advanced_sequencer_table=[TableEntry(repetition_count=program.repetition_count, element_number=1, jump_flag=0)],
        sequencer_tables=[sequencer_table],
        waveforms=tuple(waveforms.keys()),
        volatile_parameter_positions=volatile_parameter_positions
    )
