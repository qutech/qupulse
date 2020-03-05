import fractions
import functools
import sys
import weakref
from enum import Enum
from typing import Optional, Set, Tuple, Callable, Dict, Union, Any, Iterable, List, NamedTuple, cast
from collections import OrderedDict
import numpy as np
from qupulse import ChannelID
from qupulse._program._loop import Loop, make_compatible
from qupulse._program.waveforms import MultiChannelWaveform
from qupulse.hardware.awgs.features import ChannelSynchronization, AmplitudeOffsetHandling, OffsetAmplitude, \
    ProgramManagement, ActivatableChannels
from qupulse.hardware.util import voltage_to_uint16, make_combined_wave, find_positions, get_sample_times
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

# What does this mean?
assert (sys.byteorder == "little")

# Anpassen
__all__ = ["TaborDevice", "TaborChannelTuple", "TaborChannel"]


class TaborSegment:
    # __slots__ = ("channel_list", "marker_list")
    #
    # def __init__(self, channel_list: List[Optional[np.ndarray]], marker_list: List[Optional[np.ndarray]]):
    #     """
    #     :param channel_list:
    #     :param marker_list:
    #     ??? Beides Listen mit jeweils Spannungswerten zu den einzelden Channlen/Markern
    #     """
    #
    #     # TODO: check if the channel and the marker count is valid
    #
    #     # Check if at least on channel is not None
    #     ch_not_empty = False
    #     for channel in channel_list:
    #         if channel is not None:
    #             ch_not_empty = True
    #             break
    #     if ch_not_empty:
    #         raise TaborException("Empty TaborSegments are not allowed")
    #
    #     # Check if all channels that aren't None are the same length
    #     comparison_length = self.num_points
    #     for channel in channel_list:
    #         if channel is not None:
    #             comparison_length = len(channel)
    #             break
    #     for channel in channel_list:
    #         if channel is not None:
    #             if not (len(channel) == comparison_length):  # can be ignored. this case isn't possible
    #                 raise TaborException("Channel entries to have to have the same length")
    #
    #     self.channel_list = List[np.asarray(Optional[np.ndarray], dtype=np.uint16)]
    #     self.marker_list = List[np.asarray(Optional[np.ndarray], dtype=bool)]
    #     # TODO: is it possible like that. It's only possible when python works with references
    #     for channel in channel_list:
    #         self.channel_list.add(None if channel is None else np.asarray(channel, dtype=np.uint16))
    #     for marker in marker_list:
    #         self.marker_list.add(None if marker is None else np.asarray(marker, dtype=bool))
    #
    #     # Check all markers have half the length of the channels
    #     for idx, marker in enumerate(marker_list):
    #         if marker is not None and len(marker) * 2 != self.num_points:
    #             raise TaborException("Marker {} has to have half of the channels length".format(idx + 1))
    #             # +1 to match the idn of the marker channel objects
    #
    # @classmethod
    # def from_binary_segment(cls, segment_data: np.ndarray) -> "TaborSegment":
    #     pass  # TODO: to implement
    #
    # @classmethod
    # def from_binary_data(cls, date_list: List[np.ndarray]) -> "TaborSegment":
    #     channel_list: List[Optional[np.ndarray]]
    #     marker_list: List[Optional[np.ndarray]]
    #
    #     # TODO: to implement
    #
    #     return cls(channel_list, marker_list)
    #
    # def __hash__(self) -> int:
    #     # TODO: is this possible?
    #     return hash(tuple((self.channel_list, self.marker_list)))
    #
    # def __eq__(self, other: "TaborSegment"):
    #     pass  # TODO: to implement
    #
    # def data(self, data_nr: int):
    #     pass  # TODO: to implement
    #
    # @property
    # def num_points(self) -> int:
    #     """
    #     Method that returns the length of the first Channel that is not None
    #     :return: length of the first Channel that is not None
    #     """
    #     for channel in self.channel_list:
    #         if channel is not None:
    #             return len(channel)
    #
    # def get_as_binary(self) -> np.ndarray:
    #     pass  # TODO: to implement

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
            raise TaborException('Channel entries to have to have the same length')

        self.ch_a = None if ch_a is None else np.asarray(ch_a, dtype=np.uint16)
        self.ch_b = None if ch_b is None else np.asarray(ch_b, dtype=np.uint16)

        self.marker_a = None if marker_a is None else np.asarray(marker_a, dtype=bool)
        self.marker_b = None if marker_b is None else np.asarray(marker_b, dtype=bool)

        if marker_a is not None and len(marker_a) * 2 != self.num_points:
            raise TaborException('Marker A has to have half of the channels length')
        if marker_b is not None and len(marker_b) * 2 != self.num_points:
            raise TaborException('Marker A has to have half of the channels length')

    @classmethod
    def from_binary_segment(cls, segment_data: np.ndarray) -> 'TaborSegment':
        data_a = segment_data.reshape((-1, 16))[1::2, :].reshape((-1,))
        data_b = segment_data.reshape((-1, 16))[0::2, :].ravel()
        return cls.from_binary_data(data_a, data_b)

    @classmethod
    def from_binary_data(cls, data_a: np.ndarray, data_b: np.ndarray) -> 'TaborSegment':
        ch_b = data_b

        channel_mask = np.uint16(2 ** 14 - 1)
        ch_a = np.bitwise_and(data_a, channel_mask)

        marker_a_mask = np.uint16(2 ** 14)
        marker_b_mask = np.uint16(2 ** 15)
        marker_data = data_a.reshape(-1, 8)[1::2, :].reshape((-1,))

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


class TaborSequencing(Enum):
    SINGLE = 1
    ADVANCED = 2


class TaborProgram:
    def __init__(self,
                 program: Loop,
                 device_properties,
                 channels,
                 markers):
        # TODO: entspricht channels und markers den Vorgaben der device_properties + kommentare entfernen bzw. Code ersetzen

        # channel_set = frozenset(channel for channel in channels if channel is not None) | frozenset(marker
        #                                                                                            for marker in
        #                                                                                            markers if
        #                                                                                            marker is not None)

        self._program = program

        self.__waveform_mode = None
        # self._channels = tuple(channels)
        # self._markers = tuple(markers)
        # self.__used_channels = channel_set
        self.__device_properties = device_properties

        self._waveforms = List["MultiChannelWaveform"]
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
    def markers(self):
        return self._markers
        # TODO: typing

    @property
    def channels(self):
        return self._channels
        # TODO: typing

    def sampled_segments(self,
                         sample_rate: fractions.Fraction,
                         voltage_amplitude: Tuple[float, float],
                         voltage_offset: Tuple[float, float],
                         voltage_transformation: Tuple[Callable, Callable]) -> Tuple[Sequence[TaborSegment],
                                                                                     Sequence[int]]:
        sample_rate = fractions.Fraction(sample_rate, 10 ** 9)

        time_array, segment_lengths = get_sample_times(self._waveforms, sample_rate)

        if np.any(segment_lengths % 16 > 0) or np.any(segment_lengths < 192):
            raise TaborException('At least one waveform has a length that is smaller 192 or not a multiple of 16')

        # TODO: ueberpruefen
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
                return np.full_like(time, 8192, dtype=np.uint16)

        def get_marker_data(waveform: MultiChannelWaveform, time, marker):
            if self._markers[marker]:
                markerID = self._markers[marker]
                return waveform.get_sampled(channel=markerID, sample_times=time) != 0
            else:
                return np.full_like(time, False, dtype=bool)

        segments = np.empty_like(self._waveforms, dtype=TaborSegment)
        for i, waveform in enumerate(self._waveforms):
            t = time_array[:segment_lengths[i]]
            marker_time = t[::2]
            segment_a = voltage_to_data(waveform, t, 0)
            segment_b = voltage_to_data(waveform, t, 1)
            assert (len(segment_a) == len(t))
            assert (len(segment_b) == len(t))
            marker_a = get_marker_data(waveform, marker_time, 0)
            marker_b = get_marker_data(waveform, marker_time, 1)
            segments[i] = TaborSegment(ch_a=segment_a,
                                       ch_b=segment_b,
                                       marker_a=marker_a,
                                       marker_b=marker_b)
        return segments, segment_lengths

    def setup_single_sequence_mode(self) -> None:
        assert self.program.depth() == 1

        sequencer_table = []
        waveforms = OrderedDict()

        for waveform, repetition_count in ((waveform_loop.waveform.get_subset_for_channels(self.__used_channels),
                                            waveform_loop.repetition_count)
                                           for waveform_loop in self.program):
            if waveform in waveforms:
                waveform_index = waveforms[waveform]
            else:
                waveform_index = len(waveforms)
                waveforms[waveform] = waveform_index
            sequencer_table.append((repetition_count, waveform_index, 0))

        self._waveforms = tuple(waveforms.keys())
        self._sequencer_tables = [sequencer_table]
        self._advanced_sequencer_table = [(self.program.repetition_count, 1, 0)]

    def setup_advanced_sequence_mode(self) -> None:
        assert self.program.depth() > 1
        assert self.program.repetition_count == 1

        self.program.flatten_and_balance(2)

        min_seq_len = self.__device_properties['min_seq_len']
        max_seq_len = self.__device_properties['max_seq_len']

        def check_merge_with_next(program, n):
            if (program[n].repetition_count == 1 and program[n + 1].repetition_count == 1 and
                    len(program[n]) + len(program[n + 1]) < max_seq_len):
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
                    if i > 0 and check_merge_with_next(self.program, i - 1):
                        pass
                    elif i + 1 < len(self.program) and check_merge_with_next(self.program, i):
                        pass

                    # check if (partial) unrolling is possible
                    elif check_partial_unroll(self.program, i):
                        i += 1

                    # check if sequence table can be extended by unrolling a neighbor
                    elif (i > 0
                          and self.program[i - 1].repetition_count > 1
                          and len(self.program[i]) + len(self.program[i - 1]) < max_seq_len):
                        self.program[i][:0] = self.program[i - 1].copy_tree_structure()[:]
                        self.program[i - 1].repetition_count -= 1

                    elif (i + 1 < len(self.program)
                          and self.program[i + 1].repetition_count > 1
                          and len(self.program[i]) + len(self.program[i + 1]) < max_seq_len):
                        self.program[i][len(self.program[i]):] = self.program[i + 1].copy_tree_structure()[:]
                        self.program[i + 1].repetition_count -= 1

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
        waveforms = OrderedDict()
        for sequencer_table_loop in self.program:
            current_sequencer_table = []
            for waveform, repetition_count in ((waveform_loop.waveform.get_subset_for_channels(self.__used_channels),
                                                waveform_loop.repetition_count)
                                               for waveform_loop in sequencer_table_loop):
                if waveform in waveforms:
                    wf_index = waveforms[waveform]
                else:
                    wf_index = len(waveforms)
                    waveforms[waveform] = wf_index
                current_sequencer_table.append((repetition_count, wf_index, 0))

            if current_sequencer_table in sequencer_tables:
                sequence_no = sequencer_tables.index(current_sequencer_table) + 1
            else:
                sequence_no = len(sequencer_tables) + 1
                sequencer_tables.append(current_sequencer_table)

            advanced_sequencer_table.append((sequencer_table_loop.repetition_count, sequence_no, 0))

        self._advanced_sequencer_table = advanced_sequencer_table
        self._sequencer_tables = sequencer_tables
        self._waveforms = tuple(waveforms.keys())

    @property
    def program(self) -> Loop:
        return self._program

    def get_sequencer_tables(self) -> List[Tuple[int, int, int]]:
        return self._sequencer_tables

    def get_advanced_sequencer_table(self) -> List[Tuple[int, int, int]]:
        return self._advanced_sequencer_table

    @property
    def waveform_mode(self) -> str:
        return self.__waveform_mode


# TODO: How does this work?
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


# TODO: How does this work?
def with_select(function_object: Callable[["TaborChannelTuple", Any], Any]) -> Callable[["TaborChannelTuple"], Any]:
    """Asserts the channel pair is selcted when the wrapped function is called"""

    @functools.wraps(function_object)
    def selector(channel_tuple: "TaborChannelTuple", *args, **kwargs) -> Any:
        channel_tuple.select()
        return function_object(channel_tuple, *args, **kwargs)

    return selector

class PlottableProgram:
    TableEntry = NamedTuple('TableEntry', [('repetition_count', int),
                                           ('element_number', int),
                                           ('jump_flag', int)])

TaborProgramMemory = NamedTuple("TaborProgramMemory", [("waveform_to_segment", np.ndarray),
                                                       ("program", TaborProgram)])

########################################################################################################################
# Device
########################################################################################################################
# Features
# TODO: maybe implement Synchronization Feature for Tabor Devices

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
        :param mirror_addresses:  addresses of multiple device which can be controlled at once. For example you can a simulator and a real Device at once
        """

        super().__init__(device_name)
        self._instr = teawg.TEWXAwg(instr_addr, paranoia_level)
        self._mirrors = tuple(teawg.TEWXAwg(address, paranoia_level) for address in mirror_addresses)
        self._coupled = None
        self._clock_marker = [0, 0, 0, 0]  # TODO: What are clock markers used for?

        if reset:
            self.send_cmd(":RES")

        # Channel
        self._channels = [TaborChannel(i + 1, self) for i in range(4)]

        # ChannelMarker
        self._channel_marker = [TaborMarkerChannel(i + 1, self) for i in range(4)]

        # ChannelTuple

        self._channel_tuples = []
        self._channel_tuples.append(TaborChannelTuple(1, self, self.channels[0:2], self.marker_channels[0:2]))
        self._channel_tuples.append(TaborChannelTuple(2, self, self.channels[2:4], self.marker_channels[2:4]))



        if external_trigger:
            raise NotImplementedError()  # pragma: no cover

        self.initialize()  # TODO: ggf. ueberarbeiten

    def cleanup(self) -> None:
        """This will be called automatically in __del__"""
        for channel_tuple in self.channel_tuples:
            channel_tuple.cleanup() #TODO: Fehler mit dem letzten Unit Test der aber ignoriert wird

    @property
    def channels(self) -> Collection["TaborChannel"]:
        return self._channels

    @property
    def marker_channels(self) -> Collection["TaborMarkerChannel"]:
        return self._channel_marker

    @property
    def channel_tuples(self) -> Collection["TaborChannelTuple"]:
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
        for instr in self.all_devices:
            print(instr)
            instr.send_cmd(cmd_str=cmd_str, paranoia_level=paranoia_level)

    def send_query(self, query_str, query_mirrors=False) -> Any:
        if query_mirrors:
            return tuple(instr.send_query(query_str) for instr in self.all_devices)
        else:
            return self._instr.send_query(query_str)

    def send_binary_data(self, pref, bin_dat, paranoia_level=None):
        for instr in self.all_devices:
            instr.send_binary_data(pref, bin_dat=bin_dat, paranoia_level=paranoia_level)

    def download_segment_lengths(self, seg_len_list, pref=":SEGM:DATA", paranoia_level=None):
        for instr in self.all_devices:
            instr.download_segment_lengths(seg_len_list, pref=pref, paranoia_level=paranoia_level)

    def download_sequencer_table(self, seq_table, pref=":SEQ:DATA", paranoia_level=None):
        for instr in self.all_devices:
            instr.download_sequencer_table(seq_table, pref=pref, paranoia_level=paranoia_level)

    def download_adv_seq_table(self, seq_table, pref=":ASEQ:DATA", paranoia_level=None):
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
                ask_str = cmd_str + "; *OPC?; :SYST:ERR?"
            else:
                ask_str = "*OPC?; :SYST:ERR?"

            *answers, opc, error_code_msg = self._visa_inst.ask(ask_str).split(";")

            error_code, error_msg = error_code_msg.split(",")
            error_code = int(error_code)
            if error_code != 0:
                _ = self._visa_inst.ask("*CLS; *OPC?")

                if error_code == -450:
                    # query queue overflow
                    self.send_cmd(cmd_str)
                else:
                    raise RuntimeError("Cannot execute command: {}\n{}: {}".format(cmd_str, error_code, error_msg))

            assert len(answers) == 0

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
            # TODO: select Channel und Marker fehlen im Device
            self.channels[ch - 1].select()
            self.marker_channels[(ch - 1) % 2].select()
            for name, query, dtype in name_query_type_list:
                data[name].append(dtype(self.send_query(query)))
        return data

    @property
    def is_open(self) -> bool:
        return self._instr.visa_inst is not None  # pragma: no cover

    def enable(self) -> None:
        self.send_cmd(":ENAB")

    def abort(self) -> None:
        self.send_cmd(":ABOR")

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
        self.send_cmd(":INST:SEL 1")
        self.send_cmd(setup_command)
        self.send_cmd(":INST:SEL 3")
        self.send_cmd(setup_command)

    def reset(self) -> None:
        self.send_cmd(':RES')
        self._coupled = None
        self.initialize()
        for channel_tuple in self.channel_tuples:
            channel_tuple[TaborProgramManagement].clear()

    def trigger(self) -> None:
        self.send_cmd(":TRIG")

    def get_readable_device(self, simulator=True) -> teawg.TEWXAwg:
        for device in self.all_devices:
            if device.fw_ver >= 3.0:
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
class TaborOffsetAmplitude(OffsetAmplitude):
    def __init__(self, channel: "TaborChannel"):
        super().__init__()
        self._parent = channel

    @property
    def offset(self) -> float:
        return float(
            self._parent.device.send_query(":INST:SEL {channel}; :VOLT:OFFS?".format(channel=self._parent.idn)))

    @offset.setter
    def offset(self, offset: float) -> None:
        pass  # TODO: to implement

    @property
    def amplitude(self) -> float:
        coupling = self._parent.device.send_query(":INST:SEL {channel}; :OUTP:COUP?".format(channel=self._parent.idn))
        if coupling == "DC":
            return float(self._parent.device.send_query(":VOLT?"))
        elif coupling == "HV":
            return float(self._parent.device.send_query(":VOLT:HV?"))
        else:
            raise TaborException("Unknown coupling: {}".format(coupling))

    @amplitude.setter
    def amplitude(self, amplitude: float) -> None:
        self._parent.device.send_cmd(":INST:SEL {channel}; :OUTP:COUP DC".format(channel=self._parent.idn))
        self._parent.device.send_cmd(":VOLT {amp}".format(amp=amplitude))

    @property
    def amplitude_offset_handling(self) -> str:
        pass  # TODO: to implement

    @amplitude_offset_handling.setter
    def amplitude_offset_handling(self, amp_offs_handling: str) -> None:
        pass  # TODO: to implement


class TaborChannelActivatable(ActivatableChannels):
    def __init__(self, marker_channel: "TaborMarkerChannel"):
        super().__init__()
        self.parent = marker_channel

    @property
    def status(self) -> bool:
        pass  # TODO: to implement

    @status.setter
    def status(self, channel_status: bool) -> None:
        command_string = ":INST:SEL {ch_id}; :OUTP {output}".format(ch_id=self.parent.idn,
                                                                    output="ON" if channel_status else "OFF")
        self.parent.device.send_cmd(command_string)


# Implementation
class TaborChannel(AWGChannel):
    def __init__(self, idn: int, device: TaborDevice):
        super().__init__(idn)

        self._device = device

        self.add_feature(TaborOffsetAmplitude(self))

    _channel_tuple: "TaborChannelTuple"

    @property
    def device(self) -> TaborDevice:
        return self._device

    @property
    def channel_tuple(self) -> Optional[AWGChannelTuple]:
        return self._channel_tuple

    def _set_channel_tuple(self, channel_tuple) -> None:
        self._channel_tuple = channel_tuple

    def select(self) -> None:
        self.device.send_cmd(":INST:SEL {channel}".format(channel=self.idn))


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

    @property
    def programs(self) -> Set[str]:
        pass  # TODO: to implement

    @with_configuration_guard
    @with_select
    def upload(self, name: str, program: Loop, channels: Tuple[Optional[ChannelID], ...],
               markers: Tuple[Optional[ChannelID], ...], voltage_transformation: Tuple[Optional[Callable], ...],
               force: bool = False) -> None:
        """
        Upload a program to the AWG.

        The policy is to prefer amending the unknown waveforms to overwriting old ones.
        """
        if len(channels) != self.num_channels:
            raise ValueError("Channel ID not specified")
        if len(markers) != self.num_markers:
            raise ValueError("Markers not specified")
        if len(voltage_transformation) != self.num_channels:
            raise ValueError("Wrong number of voltage transformations")

        # adjust program to fit criteria
        sample_rate = self._parent.device.channel_tuples[0].sample_rate
        make_compatible(program,
                        minimal_waveform_length=192,
                        waveform_quantum=16,
                        sample_rate=fractions.Fraction(sample_rate, 10 ** 9))

    def remove(self, name: str) -> None:
        """
        Remove a program from the AWG.

        Also discards all waveforms referenced only by the program identified by name.

        Args:
            name (str): The name of the program to remove.
        """
        self._parent.free_program(name)
        self._parent.cleanup()

    def clear(self) -> None:
        """Delete all segments and clear memory"""

        self._parent.device.channels[0].select()
        self._parent.device.send_cmd(':TRAC:DEL:ALL')
        self._parent.device.send_cmd(':SOUR:SEQ:DEL:ALL')
        self._parent.device.send_cmd(':ASEQ:DEL')

        self._parent.device.send_cmd(':TRAC:DEF 1, 192')
        self._parent.device.send_cmd(':TRAC:SEL 1')
        self._parent.device.send_cmd(':TRAC:MODE COMB')
        self._parent.device.send_binary_data(pref=':TRAC:DATA', bin_dat=self._parent._idle_segment.get_as_binary())

        self._parent._segment_lengths = 192 * np.ones(1, dtype=np.uint32)
        self._parent._segment_capacity = 192 * np.ones(1, dtype=np.uint32)
        self._parent._segment_hashes = np.ones(1, dtype=np.int64) * hash(self._parent._idle_segment)
        self._parent._segment_references = np.ones(1, dtype=np.uint32)

        self._parent._advanced_sequence_table = []
        self._parent._sequencer_tables = []

        self._parent._known_programs = dict()
        self._parent.change_armed_program(None)

    @with_select
    def arm(self, name: Optional[str]) -> None:
        if self._parent._current_program == name:
            self._parent.device.send_cmd("SEQ:SEL 1")
        else:
            self._parent.change_armed_program(name)


class TaborChannelTuple(AWGChannelTuple):
    def __init__(self, idn: int, device: TaborDevice, channels: Iterable["TaborChannel"],
                 marker_channels: Iterable["TaborMarkerChannel"]):
        super().__init__(idn)
        self._device = device  # TODO: weakref.ref(device) can't be used like in  the old driver

        self._configuration_guard_count = 0
        self._is_in_config_mode = False

        self._channels = tuple(channels)
        self._marker_channels = tuple(marker_channels)

        for channel in self.channels:
            channel._set_channel_tuple(self)
        for marker_ch in self.marker_channels:
            marker_ch._set_channel_tuple(self)

        self.add_feature(TaborProgramManagement(self))

        self._idle_segment = TaborSegment(voltage_to_uint16(voltage=np.zeros(192),
                                                            output_amplitude=0.5,
                                                            output_offset=0., resolution=14),
                                          voltage_to_uint16(voltage=np.zeros(192),
                                                            output_amplitude=0.5,
                                                            output_offset=0., resolution=14), None, None)

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

    def select(self) -> None:
        # TODO: keep this?
        self.channels[0].select()

    @property
    def device(self) -> TaborDevice:
        return self._device

    @property
    def channels(self) -> Collection["TaborChannel"]:
        return self._channels


    @property
    def marker_channels(self) -> Collection["TaborMarkerChannel"]:
        return self._marker_channels

    @property
    def sample_rate(self) -> float:
        #TODO: keep sample_rate in tuple or put it in the channel
        return self.device.send_query(":INST:SEL {channel}; :FREQ:RAST?".format(channel=self.channels[0].idn))

    @property
    def total_capacity(self) -> int:
        return int(self.device.dev_properties["max_arb_mem"]) // 2

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

        old_segment = device.send_query(":TRAC:SEL?")
        waveforms = []
        uploaded_waveform_indices = np.flatnonzero(
            self._segment_references) + 1  # uploaded_waveform_indices sind gleich wie im alten Treiber

        for segment in uploaded_waveform_indices:
            device.send_cmd(':TRAC:SEL {}'.format(segment), paranoia_level=self.internal_paranoia_level)
            waveforms.append(device.read_act_seg_dat())
        device.send_cmd(':TRAC:SEL {}'.format(old_segment), paranoia_level=self.internal_paranoia_level)
        return waveforms

    @with_select
    def read_sequence_tables(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        device = self.device.get_readable_device(simulator=True)

        old_sequence = device.send_query(":SEQ:SEL?")
        sequences = []
        uploaded_sequence_indices = np.arange(len(self._sequencer_tables)) + 1
        for sequence in uploaded_sequence_indices:
            device.send_cmd(':SEQ:SEL {}'.format(sequence), paranoia_level=self.internal_paranoia_level)
            sequences.append(device.read_sequencer_table())
        device.send_cmd(':SEQ:SEL {}'.format(old_sequence), paranoia_level=self.internal_paranoia_level)
        return sequences

    @with_select
    def read_advanced_sequencer_table(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.device.get_readable_device(simulator=True).read_adv_seq_table()


    # upload im Feature

    def read_complete_program(self) -> PlottableProgram:
        return PlottableProgram.from_read_data(self.read_waveforms(),
                                               self.read_sequence_tables(),
                                               self.read_advanced_sequencer_table())

    # clear im Feature


    def _find_place_for_segments_in_memory(self, segments: Sequence, segment_lengths: Sequence) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
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

        assert len(known_pos_in_memory) == 0 or np.all(
            self._segment_hashes[known_pos_in_memory] == segment_hashes[known])

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
            raise MemoryError('Fragmentation does not allow upload.',
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

        self.device.send_cmd(':TRAC:DEF {}, {}'.format(segment_no, segment.num_points),
                             paranoia_level=self.internal_paranoia_level)
        self._segment_lengths[segment_index] = segment.num_points

        self.device.send_cmd(':TRAC:SEL {}'.format(segment_no), paranoia_level=self.internal_paranoia_level)

        self.device.send_cmd(':TRAC:MODE COMB', paranoia_level=self.internal_paranoia_level)
        wf_data = segment.get_as_binary()

        self.device.send_binary_data(pref=":TRAC:DATA", bin_dat=wf_data)
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
                self.device.send_cmd(":TRAC:DEF {},{}".format(i + 1, segment_lengths[i]))

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
                self.device.send_cmd("; ".join("TRAC:DEL {}".format(i + 1)
                                               for i in range(chunk_start, min(chunk_start + chunk_size, old_end))))
        except Exception as e:
            raise TaborUndefinedState("Error during cleanup. Device is in undefined state.", device=self) from e

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

    # arm im Feature

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

                while len(sequencer_tables[1]) < self.device.dev_properties['min_seq_len']:
                    assert advanced_sequencer_table[0][0] == 1
                    sequencer_tables[1].append((1, 1, 0))

        # insert idle sequence in advanced sequence table
        advanced_sequencer_table = [(1, 1, 1)] + advanced_sequencer_table

        while len(advanced_sequencer_table) < self.device.dev_properties['min_aseq_len']:
            advanced_sequencer_table.append((1, 1, 0))

        # reset sequencer and advanced sequencer tables to fix bug which occurs when switching between some programs
        self.device.send_cmd('SEQ:DEL:ALL')
        self._sequencer_tables = []
        self.device.send_cmd('ASEQ:DEL')
        self._advanced_sequence_table = []

        # download all sequence tables
        for i, sequencer_table in enumerate(sequencer_tables):
            self.device.send_cmd('SEQ:SEL {}'.format(i + 1))
            self.device.download_sequencer_table(sequencer_table)
        self._sequencer_tables = sequencer_tables
        self.device.send_cmd('SEQ:SEL 1')

        self.device.download_adv_seq_table(advanced_sequencer_table)
        self._advanced_sequence_table = advanced_sequencer_table

        self._current_program = name

    @with_select
    def run_current_program(self) -> None:
        if self._current_program:
            self.device.send_cmd(':TRIG', paranoia_level=self.internal_paranoia_level)
        else:
            raise RuntimeError("No program active")

    @property
    def programs(self) -> Set[str]:
        """The set of program names that can currently be executed on the hardware AWG."""
        return set(program for program in self._known_programs.keys())

    @property
    def num_channels(self) -> int:
        return len(self.channels)

    @property
    def num_markers(self) -> int:
        return len(self.marker_channels)

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
                cmd = ""
                for channel in self.channels:
                    cmd = cmd + ":INST:SEL {ch_id}; :OUTP OFF;".format(ch_id=channel.idn)
                self.device.send_cmd(cmd[:-1])

            for marker_ch in self.marker_channels:
                marker_ch[TaborMarkerChannelActivatable].status = False

            self.device.send_cmd(":SOUR:FUNC:MODE FIX")


            wf_mode_cmd = ':SOUR:FUNC:MODE FIX'

            cmd = ';'.join([out_cmd, marker_0_cmd, marker_1_cmd, wf_mode_cmd])
            self.device.send_cmd(cmd, paranoia_level=self.CONFIG_MODE_PARANOIA_LEVEL)
            self._is_in_config_mode = True

    @with_select
    def _exit_config_mode(self) -> None:
        """Leave the configuration mode. Enter advanced sequence mode and turn on all outputs"""

        # TODO: change implementation for channel synchronisation feature

        if self.device.send_query(":INST:COUP:STAT?") == "ON":
            # Coupled -> switch all channels at once
            other_channel_tuple: TaborChannelTuple
            if self.channels == self.device.channel_tuples[0].channels:
                other_channel_tuple = self.device.channel_tuples[1]
            else:
                other_channel_tuple = self.device.channel_tuples[0]

            if not other_channel_tuple._is_in_config_mode:
                self.device.send_cmd(":SOUR:FUNC:MODE ASEQ")
                self.device.send_cmd(":SEQ:SEL 1")
                self.device.send_cmd(":OUTP:ALL ON")

        else:
            self.device.send_cmd(":SOUR:FUNC:MODE ASEQ")
            self.device.send_cmd(":SEQ:SEL 1")

            cmd = ""
            for channel in self.channels:
                cmd = cmd + ":INST:SEL {}; :OUTP ON;".format(channel.idn)
            self.device.send_cmd(cmd[:-1])

        for marker_ch in self.marker_channels:
            marker_ch[TaborMarkerChannelActivatable].status = True

        self._is_in_config_mode = False


########################################################################################################################
# Marker Channel
########################################################################################################################
# Features
class TaborMarkerChannelActivatable(ActivatableChannels):
    def __init__(self, marker_channel: "TaborMarkerChannel"):
        super().__init__()
        self._parent = marker_channel

    @property
    def status(self) -> bool:
        raise NotImplementedError

    @status.setter
    def status(self, channel_state: bool) -> None:
        command_string = ":INST:SEL {channel}; :SOUR:MARK:SEL {marker}; :SOUR:MARK:SOUR USER; :SOUR:MARK:STAT {active}"
        command_string = command_string.format(
            channel=self._parent.channel_tuple.channels[0].idn,
            marker=self._parent.channel_tuple.marker_channels.index(self._parent)+1,
            active="ON" if channel_state else "OFF")
        self._parent.device.send_cmd(command_string)


# Implementation
class TaborMarkerChannel(AWGMarkerChannel):
    def __init__(self, idn: int, device: TaborDevice):
        super().__init__(idn)
        self._device = device

        self.add_feature(TaborMarkerChannelActivatable(self))

    _channel_tuple: "TaborChannelTuple"

    @property
    def device(self) -> TaborDevice:
        return self._device

    @property
    def channel_tuple(self) -> Optional[TaborChannelTuple]:
        return self._channel_tuple

    def _set_channel_tuple(self, channel_tuple) -> None:
        self._channel_tuple = channel_tuple

    def select(self) -> None:
        self.device.send_cmd(":SOUR:MARK:SEL {marker}".format(marker=self.idn))


########################################################################################################################
class TaborException(Exception):
    pass


class TaborUndefinedState(TaborException):
    """If this exception is raised the attached tabor device is in an undefined state.
    It is highly recommended to call reset it."""

    def __init__(self, *args, device: Union[TaborDevice, TaborChannelTuple]):
        super().__init__(*args)
        self.device = device

    def reset_device(self):
        if isinstance(self.device, TaborDevice):
            self.device.reset()
        elif isinstance(self.device, TaborChannelTuple):
            self.device.cleanup()
            self.device.clear()
