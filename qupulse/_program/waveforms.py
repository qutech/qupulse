"""This module contains all waveform classes

Classes:
    - Waveform: An instantiated pulse which can be sampled to a raw voltage value array.
"""

import itertools
from abc import ABCMeta, abstractmethod
from weakref import WeakValueDictionary, ref
from typing import Union, Set, Sequence, NamedTuple, Tuple, Any, Iterable, FrozenSet, Optional

import numpy as np

from qupulse import ChannelID
from qupulse.utils import checked_int_cast, isclose
from qupulse.utils.types import TimeType, time_from_float
from qupulse.comparable import Comparable
from qupulse.expressions import ExpressionScalar
from qupulse.pulses.interpolation import InterpolationStrategy
from qupulse._program.transformation import Transformation


__all__ = ["Waveform", "TableWaveform", "TableWaveformEntry", "FunctionWaveform", "SequenceWaveform",
           "MultiChannelWaveform", "RepetitionWaveform", "TransformingWaveform"]


class Waveform(Comparable, metaclass=ABCMeta):
    """Represents an instantiated PulseTemplate which can be sampled to retrieve arrays of voltage
    values for the hardware."""

    __sampled_cache = WeakValueDictionary()

    @property
    @abstractmethod
    def duration(self) -> TimeType:
        """The duration of the waveform in time units."""

    @abstractmethod
    def unsafe_sample(self,
                      channel: ChannelID,
                      sample_times: np.ndarray,
                      output_array: Union[np.ndarray, None]=None) -> np.ndarray:
        """Sample the waveform at given sample times.

        The unsafe means that there are no sanity checks performed. The provided sample times are assumed to be
        monotonously increasing and lie in the range of [0, waveform.duration]

        Args:
            sample_times: Times at which this Waveform will be sampled.
            output_array: Has to be either None or an array of the same size and type as sample_times. If
                not None, the sampled values will be written here and this array will be returned
        Result:
            The sampled values of this Waveform at the provided sample times. Has the same number of
            elements as sample_times.
        """

    def get_sampled(self,
                    channel: ChannelID,
                    sample_times: np.ndarray,
                    output_array: Union[np.ndarray, None]=None) -> np.ndarray:
        """A wrapper to the unsafe_sample method which caches the result. This method enforces the constrains
        unsafe_sample expects and caches the result to save memory.

        Args/Result:
            sample_times: Times at which this Waveform will be sampled.
            output_array: Has to be either None or an array of the same size and type as sample_times.
                If an array is given, the sampled values will be written into the given array and it will be returned.
                Otherwise, a new array will be created and cached to save memory.

        Result:
            The sampled values of this Waveform at the provided sample times.
        """
        if len(sample_times) == 0:
            if output_array is None:
                return np.zeros_like(sample_times)
            elif len(output_array) == len(sample_times):
                return output_array
            else:
                raise ValueError('Output array length and sample time length are different')

        if np.any(sample_times[:-1] >= sample_times[1:]):
            raise ValueError('The sample times are not monotonously increasing')
        if sample_times[0] < 0 or sample_times[-1] > self.duration:
            raise ValueError('The sample times are not in the range [0, duration]')
        if channel not in self.defined_channels:
            raise KeyError('Channel not defined in this waveform: {}'.format(channel))

        if output_array is None:
            # cache the result to save memory
            result = self.unsafe_sample(channel, sample_times)
            result.flags.writeable = False
            key = hash(bytes(result))
            if key not in self.__sampled_cache:
                self.__sampled_cache[key] = result
            return self.__sampled_cache[key]
        else:
            if len(output_array) != len(sample_times):
                raise ValueError('Output array length and sample time length are different')
            # use the user provided memory
            return self.unsafe_sample(channel=channel,
                                      sample_times=sample_times,
                                      output_array=output_array)

    @property
    @abstractmethod
    def defined_channels(self) -> Set[ChannelID]:
        """The channels this waveform should played on. Use
            :func:`~qupulse.pulses.instructions.get_measurement_windows` to get a waveform for a subset of these."""

    @abstractmethod
    def unsafe_get_subset_for_channels(self, channels: Set[ChannelID]) -> 'Waveform':
        """Unsafe version of :func:`~qupulse.pulses.instructions.get_measurement_windows`."""

    def get_subset_for_channels(self, channels: Set[ChannelID]) -> 'Waveform':
        """Get a waveform that only describes the channels contained in `channels`.

        Args:
            channels: A channel set the return value should confine to.

        Raises:
            KeyError: If `channels` is not a subset of the waveform's defined channels.

        Returns:
            A waveform with waveform.defined_channels == `channels`
        """
        if not channels <= self.defined_channels:
            raise KeyError('Channels not defined on waveform: {}'.format(channels))
        if channels == self.defined_channels:
            return self
        return self.unsafe_get_subset_for_channels(channels=channels)


class TableWaveformEntry(NamedTuple('TableWaveformEntry', [('t', float),
                                                           ('v', float),
                                                           ('interp', InterpolationStrategy)])):
    def __init__(self, t: float, v: float, interp: InterpolationStrategy):
        if not callable(interp):
            raise TypeError('{} is neither callable nor of type InterpolationStrategy'.format(interp))


class TableWaveform(Waveform):
    EntryInInit = Union[TableWaveformEntry, Tuple[float, float, InterpolationStrategy]]

    """Waveform obtained from instantiating a TablePulseTemplate."""
    def __init__(self,
                 channel: ChannelID,
                 waveform_table: Sequence[EntryInInit]) -> None:
        """Create a new TableWaveform instance.

        Args:
            waveform_table (ImmutableList(WaveformTableEntry)): A list of instantiated table
                entries of the form (time as float, voltage as float, interpolation strategy).
        """
        super().__init__()

        self._table = self._validate_input(waveform_table)
        self._channel_id = channel

    @staticmethod
    def _validate_input(input_waveform_table: Sequence[EntryInInit]) -> Tuple[TableWaveformEntry, ...]:
        """ Checks that:
         - the time is increasing,
         - there are at least two entries
        and removes subsequent entries with same time or voltage values.

        :param input_waveform_table:
        :return:
        """
        if len(input_waveform_table) < 2:
            raise ValueError("Waveform table has less than two entries.")

        if input_waveform_table[0][0] != 0:
            raise ValueError('First time entry is not zero.')

        if input_waveform_table[-1][0] == 0:
            raise ValueError('Last time entry is zero.')

        output_waveform_table = []

        previous_t = 0
        previous_v = None
        for (t, v, interp), (next_t, next_v, _) in itertools.zip_longest(input_waveform_table,
                                                                         input_waveform_table[1:],
                                                                         fillvalue=(float('inf'), None, None)):
            if next_t < t:
                if next_t < 0:
                    raise ValueError('Negative time values are not allowed.')
                else:
                    raise ValueError('Times are not increasing.')

            if (previous_t != t or t != next_t) and (previous_v != v or v != next_v):
                previous_t = t
                previous_v = v
                output_waveform_table.append(TableWaveformEntry(t, v, interp))

        return tuple(output_waveform_table)

    @property
    def compare_key(self) -> Any:
        return self._channel_id, self._table

    @property
    def duration(self) -> TimeType:
        return time_from_float(self._table[-1].t)

    def unsafe_sample(self,
                      channel: ChannelID,
                      sample_times: np.ndarray,
                      output_array: Union[np.ndarray, None]=None) -> np.ndarray:
        if output_array is None:
            output_array = np.empty_like(sample_times)

        for entry1, entry2 in zip(self._table[:-1], self._table[1:]):
            indices = slice(np.searchsorted(sample_times, entry1.t, 'left'),
                            np.searchsorted(sample_times, entry2.t, 'right'))
            output_array[indices] = \
                entry2.interp((entry1.t, entry1.v), (entry2.t, entry2.v), sample_times[indices])
        return output_array

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return {self._channel_id}

    def unsafe_get_subset_for_channels(self, channels: Set[ChannelID]) -> 'Waveform':
        return self


class FunctionWaveform(Waveform):
    """Waveform obtained from instantiating a FunctionPulseTemplate."""

    def __init__(self, expression: ExpressionScalar,
                 duration: float,
                 channel: ChannelID) -> None:
        """Creates a new FunctionWaveform instance.

        Args:
            expression: The function represented by this FunctionWaveform
                as a mathematical expression where 't' denotes the time variable. It must not have other variables
            duration: The duration of the waveform
            measurement_windows: A list of measurement windows
            channel: The channel this waveform is played on
        """
        super().__init__()
        if set(expression.variables) - set('t'):
            raise ValueError('FunctionWaveforms may not depend on anything but "t"')

        self._expression = expression
        self._duration = time_from_float(duration)
        self._channel_id = channel

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return {self._channel_id}

    @property
    def compare_key(self) -> Any:
        return self._channel_id, self._expression, self._duration

    @property
    def duration(self) -> TimeType:
        return self._duration

    def unsafe_sample(self,
                      channel: ChannelID,
                      sample_times: np.ndarray,
                      output_array: Union[np.ndarray, None] = None) -> np.ndarray:
        if output_array is None:
            output_array = np.empty(len(sample_times))
        output_array[:] = self._expression.evaluate_numeric(t=sample_times)
        return output_array

    def unsafe_get_subset_for_channels(self, channels: Set[ChannelID]) -> Waveform:
        return self


class SequenceWaveform(Waveform):
    """This class allows putting multiple PulseTemplate together in one waveform on the hardware."""
    def __init__(self, sub_waveforms: Iterable[Waveform]):
        """

        :param subwaveforms: All waveforms must have the same defined channels
        """
        if not sub_waveforms:
            raise ValueError(
                "SequenceWaveform cannot be constructed without channel waveforms."
            )

        def flattened_sub_waveforms() -> Iterable[Waveform]:
            for sub_waveform in sub_waveforms:
                if isinstance(sub_waveform, SequenceWaveform):
                    yield from sub_waveform._sequenced_waveforms
                else:
                    yield sub_waveform

        self._sequenced_waveforms = tuple(flattened_sub_waveforms())
        self._duration = sum(waveform.duration for waveform in self._sequenced_waveforms)
        if not all(waveform.defined_channels == self.defined_channels for waveform in self._sequenced_waveforms[1:]):
            raise ValueError(
                "SequenceWaveform cannot be constructed from waveforms of different"
                "defined channels."
            )

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return self._sequenced_waveforms[0].defined_channels

    def unsafe_sample(self,
                      channel: ChannelID,
                      sample_times: np.ndarray,
                      output_array: Union[np.ndarray, None]=None) -> np.ndarray:
        if output_array is None:
            output_array = np.empty_like(sample_times)
        time = 0
        for subwaveform in self._sequenced_waveforms:
            # before you change anything here, make sure to understand the difference between basic and advanced
            # indexing in numpy and their copy/reference behaviour
            end = time + subwaveform.duration

            indices = slice(*np.searchsorted(sample_times, (float(time), float(end)), 'left'))
            subwaveform.unsafe_sample(channel=channel,
                                      sample_times=sample_times[indices]-np.float64(time),
                                      output_array=output_array[indices])
            time = end
        return output_array

    @property
    def compare_key(self) -> Tuple[Waveform]:
        return self._sequenced_waveforms

    @property
    def duration(self) -> TimeType:
        return self._duration

    def unsafe_get_subset_for_channels(self, channels: Set[ChannelID]) -> 'Waveform':
        return SequenceWaveform(
            sub_waveform.unsafe_get_subset_for_channels(channels & sub_waveform.defined_channels)
            for sub_waveform in self._sequenced_waveforms if sub_waveform.defined_channels & channels)


class MultiChannelWaveform(Waveform):
    """A MultiChannelWaveform is a Waveform object that allows combining arbitrary Waveform objects
    to into a single waveform defined for several channels.

    The number of channels used by the MultiChannelWaveform object is the sum of the channels used
    by the Waveform objects it consists of.

    MultiChannelWaveform allows an arbitrary mapping of channels defined by the Waveforms it
    consists of and the channels it defines. For example, if the MultiChannelWaveform consists
    of a two Waveform objects A and B which define two channels each, then the channels of the
    MultiChannelWaveform may be 0: A.1, 1: B.0, 2: B.1, 3: A.0 where A.0 means channel 0 of Waveform
    object A.

    The following constraints must hold:
     - The durations of all Waveform objects must be equal.
     - The channel mapping must be sane, i.e., no channel of the MultiChannelWaveform must be
        assigned more than one channel of any Waveform object it consists of
    """

    def __init__(self, sub_waveforms: Iterable[Waveform]) -> None:
        """Create a new MultiChannelWaveform instance.

        Requires a list of subwaveforms in the form (Waveform, List(int)) where the list defines
        the channel mapping, i.e., a value y at index x in the list means that channel x of the
        subwaveform will be mapped to channel y of this MultiChannelWaveform object.

        Args:
            sub_waveforms (Iterable( Waveform )): The list of sub waveforms of this
                MultiChannelWaveform
        Raises:
            ValueError, if a channel mapping is out of bounds of the channels defined by this
                MultiChannelWaveform
            ValueError, if several subwaveform channels are assigned to a single channel of this
                MultiChannelWaveform
            ValueError, if subwaveforms have inconsistent durations
        """
        super().__init__()
        if not sub_waveforms:
            raise ValueError(
                "MultiChannelWaveform cannot be constructed without channel waveforms."
            )

        # avoid unnecessary multi channel nesting
        def flatten_sub_waveforms(to_flatten):
            for sub_waveform in to_flatten:
                if isinstance(sub_waveform, MultiChannelWaveform):
                    yield from sub_waveform._sub_waveforms
                else:
                    yield sub_waveform

        # sort the waveforms with their defined channels to make compare key reproducible
        def get_sub_waveform_sort_key(waveform):
            return tuple(sorted(tuple('{}_stringified_numeric_channel'.format(ch) if isinstance(ch, int) else ch
                                      for ch in waveform.defined_channels)))

        self._sub_waveforms = tuple(sorted(flatten_sub_waveforms(sub_waveforms),
                                           key=get_sub_waveform_sort_key))

        self.__defined_channels = set()
        for waveform in self._sub_waveforms:
            if waveform.defined_channels & self.__defined_channels:
                raise ValueError('Channel may not be defined in multiple waveforms',
                                 waveform.defined_channels & self.__defined_channels)
            self.__defined_channels |= waveform.defined_channels

        if not all(isclose(waveform.duration, self._sub_waveforms[0].duration) for waveform in self._sub_waveforms[1:]):
            # meaningful error message:
            durations = {}

            for waveform in self._sub_waveforms:
                for duration, channels in durations.items():
                    if isclose(waveform.duration, duration):
                        channels.update(waveform.defined_channels)
                        break
                else:
                    durations[waveform.duration] = set(waveform.defined_channels)

            raise ValueError(
                "MultiChannelWaveform cannot be constructed from channel waveforms of different durations.",
                durations
            )

    @property
    def duration(self) -> TimeType:
        return self._sub_waveforms[0].duration

    def __getitem__(self, key: ChannelID) -> Waveform:
        for waveform in self._sub_waveforms:
            if key in waveform.defined_channels:
                return waveform
        raise KeyError('Unknown channel ID: {}'.format(key), key)

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return self.__defined_channels

    @property
    def compare_key(self) -> Any:
        # sort with channels
        return self._sub_waveforms

    def unsafe_sample(self,
                      channel: ChannelID,
                      sample_times: np.ndarray,
                      output_array: Union[np.ndarray, None]=None) -> np.ndarray:
        return self[channel].unsafe_sample(channel, sample_times, output_array)

    def unsafe_get_subset_for_channels(self, channels: Set[ChannelID]) -> 'Waveform':
        relevant_sub_waveforms = tuple(swf for swf in self._sub_waveforms if swf.defined_channels & channels)
        if len(relevant_sub_waveforms) == 1:
            return relevant_sub_waveforms[0].get_subset_for_channels(channels)
        elif len(relevant_sub_waveforms) > 1:
            return MultiChannelWaveform(
                sub_waveform.get_subset_for_channels(channels & sub_waveform.defined_channels)
                for sub_waveform in relevant_sub_waveforms)
        else:
            raise KeyError('Unknown channels: {}'.format(channels))


class RepetitionWaveform(Waveform):
    """This class allows putting multiple PulseTemplate together in one waveform on the hardware."""
    def __init__(self, body: Waveform, repetition_count: int):
        self._body = body
        self._repetition_count = checked_int_cast(repetition_count)
        if repetition_count < 1 or not isinstance(repetition_count, int):
            raise ValueError('Repetition count must be an integer >0')

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return self._body.defined_channels

    def unsafe_sample(self,
                      channel: ChannelID,
                      sample_times: np.ndarray,
                      output_array: Union[np.ndarray, None]=None) -> np.ndarray:
        if output_array is None:
            output_array = np.empty_like(sample_times)
        body_duration = self._body.duration
        time = 0
        for _ in range(self._repetition_count):
            end = time + body_duration
            indices = slice(*np.searchsorted(sample_times, (float(time), float(end)), 'left'))
            self._body.unsafe_sample(channel=channel,
                                     sample_times=sample_times[indices] - time,
                                     output_array=output_array[indices])
            time = end
        return output_array

    @property
    def compare_key(self) -> Tuple[Any, int]:
        return self._body.compare_key, self._repetition_count

    @property
    def duration(self) -> TimeType:
        return self._body.duration*self._repetition_count

    def unsafe_get_subset_for_channels(self, channels: Set[ChannelID]) -> 'RepetitionWaveform':
        return RepetitionWaveform(body=self._body.unsafe_get_subset_for_channels(channels),
                                  repetition_count=self._repetition_count)


class TransformingWaveform(Waveform):
    def __init__(self, inner_waveform: Waveform, transformation: Transformation):
        """"""
        self._inner_waveform = inner_waveform
        self._transformation = transformation

        # cache data of inner channels based identified and invalidated by the sample times
        self._cached_data = None
        self._cached_times = lambda: None

    @property
    def inner_waveform(self) -> Waveform:
        return self._inner_waveform

    @property
    def transformation(self) -> Transformation:
        return self._transformation

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return self.transformation.get_output_channels(self.inner_waveform.defined_channels)

    @property
    def compare_key(self) -> Tuple[Waveform, Transformation]:
        return self.inner_waveform, self.transformation

    @property
    def duration(self) -> TimeType:
        return self.inner_waveform.duration

    def unsafe_get_subset_for_channels(self, channels: Set[ChannelID]) -> 'SubsetWaveform':
        return SubsetWaveform(self, channel_subset=channels)

    def unsafe_sample(self,
                      channel: ChannelID,
                      sample_times: np.ndarray,
                      output_array: Union[np.ndarray, None] = None) -> np.ndarray:
        if self._cached_times() is not sample_times:
            self._cached_data = dict()
            self._cached_times = ref(sample_times)

        if channel not in self._cached_data:

            inner_channels = self.transformation.get_input_channels({channel})

            inner_data = {inner_channel: self.inner_waveform.unsafe_sample(inner_channel, sample_times)
                          for inner_channel in inner_channels}

            outer_data = self.transformation(sample_times, inner_data)
            self._cached_data.update(outer_data)

        if output_array is None:
            output_array = self._cached_data[channel]
        else:
            output_array[:] = self._cached_data[channel]

        return output_array


class SubsetWaveform(Waveform):
    def __init__(self, inner_waveform: Waveform, channel_subset: Set[ChannelID]):
        self._inner_waveform = inner_waveform
        self._channel_subset = frozenset(channel_subset)

    @property
    def inner_waveform(self) -> Waveform:
        return self._inner_waveform

    @property
    def defined_channels(self) -> FrozenSet[ChannelID]:
        return self._channel_subset

    @property
    def duration(self) -> TimeType:
        return self.inner_waveform.duration

    @property
    def compare_key(self) -> Tuple[frozenset, Waveform]:
        return self.defined_channels, self.inner_waveform

    def unsafe_get_subset_for_channels(self, channels: Set[ChannelID]) -> Waveform:
        return self.inner_waveform.get_subset_for_channels(channels)

    def unsafe_sample(self,
                      channel: ChannelID,
                      sample_times: np.ndarray,
                      output_array: Union[np.ndarray, None]=None) -> np.ndarray:
        return self.inner_waveform.unsafe_sample(channel, sample_times, output_array)
