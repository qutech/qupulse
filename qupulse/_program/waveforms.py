"""This module contains all waveform classes

Classes:
    - Waveform: An instantiated pulse which can be sampled to a raw voltage value array.
"""

import itertools
from abc import ABCMeta, abstractmethod
from weakref import WeakValueDictionary, ref
from typing import Union, Set, Sequence, NamedTuple, Tuple, Any, Iterable, FrozenSet, Optional, Mapping, AbstractSet
import operator
import collections

import numpy as np

from qupulse import ChannelID
from qupulse.utils import checked_int_cast, isclose
from qupulse.utils.types import TimeType, FrozenDict
from qupulse.utils.numeric import are_durations_compatible
from qupulse.comparable import Comparable
from qupulse.expressions import ExpressionScalar
from qupulse.pulses.interpolation import InterpolationStrategy
from qupulse._program.transformation import Transformation


__all__ = ["Waveform", "TableWaveform", "TableWaveformEntry", "FunctionWaveform", "SequenceWaveform",
           "MultiChannelWaveform", "RepetitionWaveform", "TransformingWaveform", "ArithmeticWaveform"]


def alloc_for_sample(size: int) -> np.ndarray:
    """All "preallocation" happens via this function. It uses NaN by default to make incomplete initialization better
    visible."""
    return np.full(shape=size, fill_value=np.nan)


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

        Args:
            sample_times: Times at which this Waveform will be sampled.
            output_array: Has to be either None or an array of the same size and type as sample_times. If an array is
                given, the sampled values will be written into the given array and it will be returned. Otherwise, a new
                array will be created and cached to save memory.

        Result:
            The sampled values of this Waveform at the provided sample times. Is `output_array` if provided
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
        if sample_times[0] < 0 or sample_times[-1] > float(self.duration):
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
    def unsafe_get_subset_for_channels(self, channels: AbstractSet[ChannelID]) -> 'Waveform':
        """Unsafe version of :func:`~qupulse.pulses.instructions.get_measurement_windows`."""

    def get_subset_for_channels(self, channels: AbstractSet[ChannelID]) -> 'Waveform':
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

    def __neg__(self):
        return FunctorWaveform(self, {ch: np.negative for ch in self.defined_channels})

    def __pos__(self):
        return self

    def last_value(self, channel) -> float:
        """Get the last value of the waveform"""
        # TODO: Optimize this
        return self.unsafe_sample(channel, np.array([float(self.duration)]))[0]


class TableWaveformEntry(NamedTuple('TableWaveformEntry', [('t', float),
                                                           ('v', float),
                                                           ('interp', InterpolationStrategy)])):
    def __init__(self, t: float, v: float, interp: Optional[InterpolationStrategy]):
        if not callable(interp) or interp is None:
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
        return TimeType.from_float(self._table[-1].t)

    def unsafe_sample(self,
                      channel: ChannelID,
                      sample_times: np.ndarray,
                      output_array: Union[np.ndarray, None]=None) -> np.ndarray:
        if output_array is None:
            output_array = alloc_for_sample(sample_times.size)

        for entry1, entry2 in zip(self._table[:-1], self._table[1:]):
            indices = slice(np.searchsorted(sample_times, entry1.t, 'left'),
                            np.searchsorted(sample_times, entry2.t, 'right'))
            output_array[indices] = \
                entry2.interp((entry1.t, entry1.v), (entry2.t, entry2.v), sample_times[indices])
        return output_array

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return {self._channel_id}

    def unsafe_get_subset_for_channels(self, channels: AbstractSet[ChannelID]) -> 'Waveform':
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
        self._duration = TimeType.from_float(duration)
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
            output_array = alloc_for_sample(sample_times.size)
        output_array[:] = self._expression.evaluate_numeric(t=sample_times)
        return output_array

    def unsafe_get_subset_for_channels(self, channels: AbstractSet[ChannelID]) -> Waveform:
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
            output_array = alloc_for_sample(sample_times.size)
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

    def unsafe_get_subset_for_channels(self, channels: AbstractSet[ChannelID]) -> 'Waveform':
        return SequenceWaveform(
            sub_waveform.unsafe_get_subset_for_channels(channels & sub_waveform.defined_channels)
            for sub_waveform in self._sequenced_waveforms if sub_waveform.defined_channels & channels)


class MultiChannelWaveform(Waveform):
    """A MultiChannelWaveform is a Waveform object that allows combining arbitrary Waveform objects
    to into a single waveform defined for several channels. Most of the time you want to use
    :py:meth:`MultiChannelWaveform.from_iterable` to construct a MultiChannelWaveform.

    Automatic padding and truncation:
    The duration of the overall waveform is specified by the `duration` argument (None means maximum sub-waveform
    duration). All channels that are not in `pad_values` need to be compatible with this duration (determined with
    :func:`are_durations_compatible`). Channels that are in `pad_values` are truncated or padded with
    the specified value to the required duration. A `None` value is replaced by the result of
    `sub_waveform.last_sample(channel_id)`.

    Implementation detail:
    Channels that have compatible durations are handled as if their pad_value entry is None. This is only relevant
    in numeric corner cases to be always well behaved.
    """

    def __init__(self,
                 sub_waveforms: Mapping[ChannelID, Waveform],
                 pad_values: Mapping[ChannelID, Optional[float]],
                 duration: TimeType) -> None:
        super().__init__()
        assert sub_waveforms

        wf_pad_dict = {}
        for ch, waveform in sub_waveforms.items():
            assert ch in waveform.defined_channels

            if ch not in pad_values:
                assert waveform.duration > duration or are_durations_compatible(duration, waveform.duration)

            # add default pad that is only required in corner cases of numeric accuracy
            pad_value = pad_values.get(ch, None)
            if pad_value is None:
                pad_value = waveform.last_value(ch)

            wf_pad_dict[ch] = (waveform, pad_value)

        self._wf_pad = FrozenDict(wf_pad_dict)
        self._duration = duration

    @classmethod
    def from_iterable(cls,
                      sub_waveforms: Iterable[Waveform],
                      pad_values: Optional[Mapping[ChannelID, Optional[float]]] = None,
                      duration: Optional[TimeType] = None
                      ) -> 'MultiChannelWaveform':
        """Construct a MultiChannelWaveform from an iterable of Waveforms.
        Args:
            sub_waveforms (Iterable( Waveform )): The list of sub waveforms of this
                MultiChannelWaveform
            pad_values: Value for padding if desired. None implies :py:meth:`Waveform.last_value`. Channels not
                mentioned must have a longer or compatible duration.
            duration: Duration of this waveform. None implies the maximum subwaveform duration.
        Raises:
            ValueError, if `sub_waveforms` is empty
            ValueError, if the defined channels several subwaveform overlap
            ValueError, if subwaveforms have incompatible durations and are not padded
            ValueError, if a channel is padded that is not defined in a subwaveform
        """
        if not sub_waveforms:
            raise ValueError(
                "MultiChannelWaveform cannot be constructed without channel waveforms."
            )
        if pad_values is None:
            pad_values = {}

        duration = max(sub_waveform.duration for sub_waveform in sub_waveforms) if duration is None else duration
        defined_channels = collections.Counter()

        flattened_wf = {}
        flattened_pad = {}

        incompatible_durations = {}
        for waveform in sub_waveforms:
            # if pad is not defined the sub waveform duration needs to be compatible with the overall duration
            undefined_pad = waveform.defined_channels - pad_values.keys()
            if waveform.duration > duration:
                # truncation is allowed
                pass
            elif undefined_pad and not are_durations_compatible(duration, waveform.duration):
                # prepare error message
                incompatible_durations.setdefault(waveform.duration, set()).update(undefined_pad)

            defined_channels.update(waveform.defined_channels)

            if isinstance(waveform, MultiChannelWaveform) and are_durations_compatible(waveform.duration, duration):
                for ch, (wf, pad) in waveform._wf_pad.items():
                    flattened_wf[ch] = wf
                    flattened_pad[ch] = pad
            else:
                for ch in waveform.defined_channels:
                    flattened_wf[ch] = waveform

        if incompatible_durations:
            raise ValueError(
                "MultiChannelWaveform cannot be constructed from channel waveforms of incompatible durations.",
                incompatible_durations
            )
        if defined_channels.most_common()[0][1] > 1:
            multi_defined = {ch for ch, count in defined_channels.items() if count > 1}
            raise ValueError('Channel may not be defined in multiple waveforms',
                             multi_defined)
        if pad_values.keys() - defined_channels.keys():
            raise ValueError('pad_values contains channels not defined in subwaveforms',
                             pad_values.keys() - defined_channels.keys())

        return cls(flattened_wf,
                   {**pad_values, **flattened_pad},
                   duration)

    @property
    def duration(self) -> TimeType:
        return self._duration

    def __getitem__(self, key: ChannelID) -> Waveform:
        try:
            return self._wf_pad[key][0]
        except KeyError:
            raise KeyError('Unknown channel ID: {}'.format(key), key)

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return self._wf_pad.keys()

    @property
    def compare_key(self) -> Any:
        return self._duration, self._wf_pad

    def unsafe_sample(self,
                      channel: ChannelID,
                      sample_times: np.ndarray,
                      output_array: Union[np.ndarray, None]=None) -> np.ndarray:
        """Pad with last value to length of longest waveform"""
        sub_waveform, pad_value = self._wf_pad[channel]
        max_idx = np.searchsorted(sample_times, float(sub_waveform.duration), 'right')
        if max_idx < len(sample_times):
            # we need to pad in the output
            if output_array is None:
                output_array = alloc_for_sample(sample_times.size)
            inner_output_array = output_array[:max_idx]

            sub_waveform.unsafe_sample(channel, sample_times, output_array=inner_output_array)
            output_array[max_idx:] = pad_value
            return output_array

        else:
            return sub_waveform.unsafe_sample(channel, sample_times, output_array=output_array)

    def unsafe_get_subset_for_channels(self, channels: AbstractSet[ChannelID]) -> 'Waveform':
        # TODO: is the optimization to detect if the result can be expressed as a sub-waveform worth it?
        # need to check duration compatibility then for consistent padding / truncation
        self_duration = self.duration
        waveforms = {}
        pad_values = {}
        padding = False

        for ch in channels:
            wf, pad = self._wf_pad[ch]
            padding = padding or wf.duration != self_duration
            waveforms[ch] = wf
            pad_values[ch] = pad

        if not padding:
            single_waveform = None
            if len(waveforms) == 1:
                single_waveform, = waveforms.values()
            elif len(set(waveforms.values())) == 1:
                _, single_waveform = waveforms.popitem()
            if single_waveform is not None:
                return single_waveform.get_subset_for_channels(channels)

        return MultiChannelWaveform(
            {ch: self._wf_pad[ch][0] for ch in channels},
            {ch: self._wf_pad[ch][1] for ch in channels},
            self._duration
        )

    def __repr__(self):
        sub_waveforms = {ch: wf for ch, (wf, _) in self._wf_pad.items()}
        pad_values = {ch: pad for ch, (_, pad) in self._wf_pad.items()}
        duration = self.duration
        return f"{type(self).__name__}(sub_waveforms={sub_waveforms}, pad_values={pad_values}, duration={duration})"


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
            output_array = alloc_for_sample(sample_times.size)
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
        return self._body.duration * self._repetition_count

    def unsafe_get_subset_for_channels(self, channels: AbstractSet[ChannelID]) -> 'RepetitionWaveform':
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


class ArithmeticWaveform(Waveform):
    """Channels only present in one waveform have the operations neutral element on the other."""

    numpy_operator_map = {'+': np.add,
                          '-': np.subtract}
    operator_map = {'+': operator.add,
                    '-': operator.sub}

    rhs_only_map = {'+': operator.pos,
                    '-': operator.neg}
    numpy_rhs_only_map = {'+': np.positive,
                          '-': np.negative}

    def __init__(self,
                 lhs: Waveform,
                 arithmetic_operator: str,
                 rhs: Waveform):
        super().__init__()
        self._lhs = lhs
        self._rhs = rhs
        self._arithmetic_operator = arithmetic_operator

        assert np.isclose(float(self._lhs.duration), float(self._rhs.duration))
        assert arithmetic_operator in self.operator_map

    @property
    def lhs(self) -> Waveform:
        return self._lhs

    @property
    def rhs(self) -> Waveform:
        return self._rhs

    @property
    def arithmetic_operator(self) -> str:
        return self._arithmetic_operator

    @property
    def duration(self) -> TimeType:
        return self._lhs.duration

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return set.union(self._lhs.defined_channels, self._rhs.defined_channels)

    def unsafe_sample(self,
                      channel: ChannelID,
                      sample_times: np.ndarray,
                      output_array: Union[np.ndarray, None] = None) -> np.ndarray:
        if channel in self._lhs.defined_channels:
            lhs = self._lhs.unsafe_sample(channel=channel, sample_times=sample_times, output_array=output_array)
        else:
            lhs = None

        if channel in self._rhs.defined_channels:
            rhs = self._rhs.unsafe_sample(channel=channel, sample_times=sample_times,
                                          output_array=None if lhs is not None else output_array)
        else:
            rhs = None

        if rhs is not None and lhs is not None:
            arithmetic_operator = self.numpy_operator_map[self._arithmetic_operator]
            if output_array is None:
                output_array = lhs
            return arithmetic_operator(lhs, rhs, out=output_array)

        else:
            if lhs is None:
                assert rhs is not None, "channel %r not in defined channels (internal bug)" % channel
                return self.numpy_rhs_only_map[self._arithmetic_operator](rhs, out=output_array)
            else:
                return lhs

    def unsafe_get_subset_for_channels(self, channels: Set[ChannelID]) -> Waveform:
        # TODO: optimization possible
        return SubsetWaveform(self, channels)

    @property
    def compare_key(self) -> Tuple[str, Waveform, Waveform]:
        return self._arithmetic_operator, self._lhs, self._rhs


class FunctorWaveform(Waveform):
    """Apply a channel wise functor that works inplace to all results"""
    def __init__(self, inner_waveform: Waveform, functor: Mapping[ChannelID, 'Callable']):
        self._inner_waveform = inner_waveform
        self._functor = dict(functor.items())

        assert set(functor.keys()) == inner_waveform.defined_channels, ("There is no default identity mapping (yet)."
                                                                        "File an issue on github if you need it.")

    @property
    def duration(self) -> TimeType:
        return self._inner_waveform.duration

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return self._inner_waveform.defined_channels

    def unsafe_sample(self,
                      channel: ChannelID,
                      sample_times: np.ndarray,
                      output_array: Union[np.ndarray, None] = None) -> np.ndarray:
        return self._functor[channel](self._inner_waveform.unsafe_sample(channel, sample_times, output_array))

    def unsafe_get_subset_for_channels(self, channels: Set[ChannelID]) -> Waveform:
        return SubsetWaveform(self, channels)

    @property
    def compare_key(self) -> Tuple[Waveform, FrozenSet]:
        return self._inner_waveform, frozenset(self._functor.items())
