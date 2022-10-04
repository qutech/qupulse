"""This module contains all waveform classes

Classes:
    - Waveform: An instantiated pulse which can be sampled to a raw voltage value array.
"""

import itertools
import operator
import warnings
from abc import ABCMeta, abstractmethod
from numbers import Real
from typing import (
    AbstractSet, Any, FrozenSet, Iterable, Mapping, NamedTuple, Sequence, Set,
    Tuple, Union, cast, Optional, List, Hashable)
from weakref import WeakValueDictionary, ref

import numpy as np

from qupulse import ChannelID
from qupulse._program.transformation import Transformation
from qupulse.utils import checked_int_cast, isclose
from qupulse.utils.types import TimeType, time_from_float
from qupulse.utils.performance import is_monotonic
from qupulse.comparable import Comparable
from qupulse.expressions import ExpressionScalar
from qupulse.pulses.interpolation import InterpolationStrategy
from qupulse.utils import checked_int_cast, isclose
from qupulse.utils.types import TimeType, time_from_float, FrozenDict
from qupulse._program.transformation import Transformation
from qupulse.utils import pairwise

class ConstantFunctionPulseTemplateWarning(UserWarning):
    """  This warning indicates a constant waveform is constructed from a FunctionPulseTemplate """
    pass

__all__ = ["Waveform", "TableWaveform", "TableWaveformEntry", "FunctionWaveform", "SequenceWaveform",
           "MultiChannelWaveform", "RepetitionWaveform", "TransformingWaveform", "ArithmeticWaveform",
           "ConstantFunctionPulseTemplateWarning"]

PULSE_TO_WAVEFORM_ERROR = None  # error margin in pulse template to waveform conversion

#  these are private because there probably will be changes here
_ALLOCATION_FUNCTION = np.full_like  # pre_allocated = ALLOCATION_FUNCTION(sample_times, **ALLOCATION_FUNCTION_KWARGS)
_ALLOCATION_FUNCTION_KWARGS = dict(fill_value=np.nan, dtype=float)


def _to_time_type(duration: Real) -> TimeType:
    if isinstance(duration, TimeType):
        return duration
    else:
        return time_from_float(float(duration), absolute_error=PULSE_TO_WAVEFORM_ERROR)


class Waveform(Comparable, metaclass=ABCMeta):
    """Represents an instantiated PulseTemplate which can be sampled to retrieve arrays of voltage
    values for the hardware."""

    __sampled_cache = WeakValueDictionary()

    __slots__ = ('_duration',)

    def __init__(self, duration: TimeType):
        self._duration = duration

    @property
    def duration(self) -> TimeType:
        """The duration of the waveform in time units."""
        return self._duration

    @abstractmethod
    def unsafe_sample(self,
                      channel: ChannelID,
                      sample_times: np.ndarray,
                      output_array: Union[np.ndarray, None] = None) -> np.ndarray:
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
                    output_array: Union[np.ndarray, None] = None) -> np.ndarray:
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
                return np.zeros_like(sample_times, dtype=float)
            elif len(output_array) == len(sample_times):
                return output_array
            else:
                raise ValueError('Output array length and sample time length are different')

        if not is_monotonic(sample_times):
            raise ValueError('The sample times are not monotonously increasing')
        if sample_times[0] < 0 or sample_times[-1] > float(self.duration):
            raise ValueError(f'The sample times [{sample_times[0]}, ..., {sample_times[-1]}] are not in the range'
                             f' [0, duration={float(self.duration)}]')
        if channel not in self.defined_channels:
            raise KeyError('Channel not defined in this waveform: {}'.format(channel))

        constant_value = self.constant_value(channel)
        if constant_value is None:
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
        else:
            if output_array is None:
                output_array = np.full_like(sample_times, fill_value=constant_value, dtype=float)
            else:
                output_array[:] = constant_value
            return output_array

    @property
    @abstractmethod
    def defined_channels(self) -> AbstractSet[ChannelID]:
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

    def is_constant(self) -> bool:
        """Convenience function to check if all channels are constant. The result is equal to
        `all(waveform.constant_value(ch) is not None for ch in waveform.defined_channels)` but might be more performant.

        Returns:
            True if all channels have constant values.
        """
        return self.constant_value_dict() is not None

    def constant_value_dict(self) -> Optional[Mapping[ChannelID, float]]:
        result = {ch: self.constant_value(ch) for ch in self.defined_channels}
        if None in result.values():
            return None
        else:
            return result

    def constant_value(self, channel: ChannelID) -> Optional[float]:
        """Checks if the requested channel has a constant value and returns it if so.

        Guarantee that this assertion passes for every t in waveform duration:
        >>> assert waveform.constant_value(channel) is None or waveform.constant_value(t) = waveform.get_sampled(channel, t)

        Args:
            channel: The channel to check

        Returns:
            None if there is no guarantee that the channel is constant. The value otherwise.
        """
        return None

    def __neg__(self):
        return FunctorWaveform.from_functor(self, {ch: np.negative for ch in self.defined_channels})

    def __pos__(self):
        return self

    def _sort_key_for_channels(self) -> Sequence[Tuple[str, int]]:
        """Makes reproducible sorting by defined channels possible"""
        return sorted((ch, 0) if isinstance(ch, str) else ('', ch) for ch in self.defined_channels)

    def reversed(self) -> 'Waveform':
        """Returns a reversed version of this waveform."""
        # We don't check for constness here because const waveforms are supposed to override this method
        return ReversedWaveform(self)


class TableWaveformEntry(NamedTuple('TableWaveformEntry', [('t', Real),
                                                           ('v', float),
                                                           ('interp', InterpolationStrategy)])):
    def __init__(self, t: float, v: float, interp: InterpolationStrategy):
        if not callable(interp):
            raise TypeError('{} is neither callable nor of type InterpolationStrategy'.format(interp))

    def __repr__(self):
        return f'{type(self).__name__}(t={self.t!r}, v={self.v!r}, interp={self.interp!r})'


class TableWaveform(Waveform):
    EntryInInit = Union[TableWaveformEntry, Tuple[float, float, InterpolationStrategy]]

    """Waveform obtained from instantiating a TablePulseTemplate."""

    __slots__ = ('_table', '_channel_id')

    def __init__(self,
                 channel: ChannelID,
                 waveform_table: Tuple[TableWaveformEntry, ...]) -> None:
        """Create a new TableWaveform instance.

        Args:
            waveform_table: A tuple of instantiated and validated table entries
        """
        if not isinstance(waveform_table, tuple):
            warnings.warn("Please use a tuple of TableWaveformEntry to construct TableWaveform directly",
                          category=DeprecationWarning)
            waveform_table = self._validate_input(waveform_table)

        super().__init__(duration=_to_time_type(waveform_table[-1].t))

        self._table = waveform_table
        self._channel_id = channel

    @staticmethod
    def _validate_input(input_waveform_table: Sequence[EntryInInit]) -> Union[Tuple[Real, Real],
                                                                              List[TableWaveformEntry]]:
        """ Checks that:
         - the time is increasing,
         - there are at least two entries

        Optimizations:
          - removes subsequent entries with same time or voltage values.
          - checks if the complete waveform is constant. Returns a (duration, value) tuple if this is the case

        Raises:
            ValueError:
              - there are less than two entries
              - the entries are not ordered in time
              - Any time is negative
              - The total length is zero

        Returns:
            A list of de-duplicated table entries
            OR
            A (duration, value) tuple if the waveform is constant
        """
        # we use an iterator here to avoid duplicate work and be maximally efficient for short tables
        # We never use StopIteration to abort iteration. It always signifies an error.
        input_iter = iter(input_waveform_table)
        try:
            first_t, first_v, first_interp = next(input_iter)
        except StopIteration:
            raise ValueError("Waveform table mut not be empty")

        if first_t != 0.0:
            raise ValueError('First time entry is not zero.')

        previous_t = 0.0
        previous_v = first_v
        output_waveform_table = [TableWaveformEntry(0.0, first_v, first_interp)]

        try:
            t, v, interp = next(input_iter)
        except StopIteration:
            raise ValueError("Waveform table has less than two entries.")
        if t < 0:
            raise ValueError('Negative time values are not allowed.')

        # constant_v is None <=> the waveform is constant until up to the current entry
        constant_v = first_interp.constant_value((previous_t, previous_v), (t, v))

        for next_t, next_v, next_interp in input_iter:
            if next_t < t:
                if next_t < 0:
                    raise ValueError('Negative time values are not allowed.')
                else:
                    raise ValueError('Times are not increasing.')

            if constant_v is not None and interp.constant_value((t, v), (next_t, next_v)) != constant_v:
                constant_v = None

            if (previous_t != t or t != next_t) and (previous_v != v or v != next_v):
                # the time and the value differ both either from the next or the previous
                # otherwise we skip the entry
                previous_t = t
                previous_v = v
                output_waveform_table.append(TableWaveformEntry(t, v, interp))

            t, v, interp = next_t, next_v, next_interp

        # Until now, we only checked that the time does not decrease. We require an increase because duration == 0
        # waveforms are ill-formed. t is now the time of the last entry.
        if t == 0:
            raise ValueError('Last time entry is zero.')

        if constant_v is not None:
            # the waveform is constant
            return t, constant_v
        else:
            # we must still add the last entry to the table
            output_waveform_table.append(TableWaveformEntry(t, v, interp))
            return output_waveform_table

    def is_constant(self) -> bool:
        # only correct if `from_table` is used
        return False

    def constant_value_dict(self) -> Optional[Mapping[ChannelID, float]]:
        # only correct if `from_table` is used
        return None

    @classmethod
    def from_table(cls, channel: ChannelID, table: Sequence[EntryInInit]) -> Union['TableWaveform', 'ConstantWaveform']:
        table = cls._validate_input(table)
        if isinstance(table, tuple):
            duration, amplitude = table
            return ConstantWaveform(duration=duration, amplitude=amplitude, channel=channel)
        else:
            return TableWaveform(channel, tuple(table))

    @property
    def compare_key(self) -> Any:
        return self._channel_id, self._table

    def unsafe_sample(self,
                      channel: ChannelID,
                      sample_times: np.ndarray,
                      output_array: Union[np.ndarray, None] = None) -> np.ndarray:
        if output_array is None:
            output_array = _ALLOCATION_FUNCTION(sample_times, **_ALLOCATION_FUNCTION_KWARGS)

        if PULSE_TO_WAVEFORM_ERROR:
            # we need to replace the last entry's t with self.duration
            *entries, last = self._table
            entries.append(TableWaveformEntry(float(self.duration), last.v, last.interp))
        else:
            entries = self._table

        for entry1, entry2 in pairwise(entries):
            indices = slice(np.searchsorted(sample_times, entry1.t, 'left'),
                            np.searchsorted(sample_times, entry2.t, 'right'))
            output_array[indices] = \
                entry2.interp((float(entry1.t), entry1.v),
                              (float(entry2.t), entry2.v),
                              sample_times[indices])
        return output_array

    @property
    def defined_channels(self) -> AbstractSet[ChannelID]:
        return {self._channel_id}

    def unsafe_get_subset_for_channels(self, channels: AbstractSet[ChannelID]) -> 'Waveform':
        return self

    def __repr__(self):
        return f'{type(self).__name__}(channel={self._channel_id!r}, waveform_table={self._table!r})'


class ConstantWaveform(Waveform):

    # TODO: remove
    _is_constant_waveform = True

    __slots__ = ('_amplitude', '_channel')

    def __init__(self, duration: Real, amplitude: Any, channel: ChannelID):
        """ Create a qupulse waveform corresponding to a ConstantPulseTemplate """
        super().__init__(duration=_to_time_type(duration))
        self._amplitude = amplitude
        self._channel = channel

    @classmethod
    def from_mapping(cls, duration: Real, constant_values: Mapping[ChannelID, float]) -> Union['ConstantWaveform',
                                                                                               'MultiChannelWaveform']:
        """Construct a ConstantWaveform or a MultiChannelWaveform of ConstantWaveforms with given duration and values"""
        assert constant_values
        duration = _to_time_type(duration)
        if len(constant_values) == 1:
            (channel, amplitude), = constant_values.items()
            return cls(duration, amplitude=amplitude, channel=channel)
        else:
            return MultiChannelWaveform([cls(duration, amplitude=amplitude, channel=channel)
                                         for channel, amplitude in constant_values.items()])

    def is_constant(self) -> bool:
        return True

    def constant_value(self, channel: ChannelID) -> Optional[float]:
        assert channel == self._channel
        return self._amplitude

    def constant_value_dict(self) -> Optional[Mapping[ChannelID, float]]:
        return {self._channel: self._amplitude}

    @property
    def defined_channels(self) -> AbstractSet[ChannelID]:
        """The channels this waveform should played on. Use
            :func:`~qupulse.pulses.instructions.get_measurement_windows` to get a waveform for a subset of these."""

        return {self._channel}

    @property
    def compare_key(self) -> Tuple[Any, ...]:
        return self._duration, self._amplitude, self._channel

    def unsafe_sample(self,
                      channel: ChannelID,
                      sample_times: np.ndarray,
                      output_array: Union[np.ndarray, None] = None) -> np.ndarray:
        if output_array is None:
            return np.full_like(sample_times, fill_value=self._amplitude, dtype=float)
        else:
            output_array[:] = self._amplitude
            return output_array

    def unsafe_get_subset_for_channels(self, channels: Set[ChannelID]) -> Waveform:
        """Unsafe version of :func:`~qupulse.pulses.instructions.get_measurement_windows`."""
        return self

    def __repr__(self):
        return f"{type(self).__name__}(duration={self.duration!r}, "\
               f"amplitude={self._amplitude!r}, channel={self._channel!r})"

    def reversed(self) -> 'Waveform':
        return self


class FunctionWaveform(Waveform):
    """Waveform obtained from instantiating a FunctionPulseTemplate."""

    __slots__ = ('_expression', '_channel_id')

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

        if set(expression.variables) - set('t'):
            raise ValueError('FunctionWaveforms may not depend on anything but "t"')
        elif not expression.variables:
            warnings.warn("Constant FunctionWaveform is not recommended as the constant propagation will be suboptimal",
                          category=ConstantFunctionPulseTemplateWarning)
        super().__init__(duration=_to_time_type(duration))
        self._expression = expression
        self._channel_id = channel

    @classmethod
    def from_expression(cls, expression: ExpressionScalar, duration: float, channel: ChannelID) -> Union['FunctionWaveform', ConstantWaveform]:
        if expression.variables:
            return cls(expression, duration, channel)
        else:
            return ConstantWaveform(amplitude=expression.evaluate_numeric(), duration=duration, channel=channel)

    def is_constant(self) -> bool:
        # only correct if `from_expression` is used
        return False

    def constant_value_dict(self) -> Optional[Mapping[ChannelID, float]]:
        # only correct if `from_expression` is used
        return None

    @property
    def defined_channels(self) -> AbstractSet[ChannelID]:
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
        evaluated = self._expression.evaluate_numeric(t=sample_times)
        if output_array is None:
            if self._expression.variables:
                return evaluated.astype(float)
            else:
                return np.full_like(sample_times, fill_value=float(evaluated), dtype=float)
        else:
            output_array[:] = evaluated
            return output_array

    def unsafe_get_subset_for_channels(self, channels: AbstractSet[ChannelID]) -> Waveform:
        return self

    def __repr__(self):
        return f"{type(self).__name__}(duration={self.duration!r}, "\
               f"expression={self._expression!r}, channel={self._channel_id!r})"


class SequenceWaveform(Waveform):
    """This class allows putting multiple PulseTemplate together in one waveform on the hardware."""

    __slots__ = ('_sequenced_waveforms', )

    def __init__(self, sub_waveforms: Iterable[Waveform]):
        """Use Waveform.from_sequence for optimal construction

        :param subwaveforms: All waveforms must have the same defined channels
        """
        if not sub_waveforms:
            raise ValueError(
                "SequenceWaveform cannot be constructed without channel waveforms."
            )

        # do not fail on iterators although we do not allow them as an argument
        sequenced_waveforms = tuple(sub_waveforms)

        super().__init__(duration=sum(waveform.duration for waveform in sequenced_waveforms))
        self._sequenced_waveforms = sequenced_waveforms

        defined_channels = self._sequenced_waveforms[0].defined_channels
        if not all(waveform.defined_channels == defined_channels
                   for waveform in itertools.islice(self._sequenced_waveforms, 1, None)):
            for waveform in self._sequenced_waveforms[1:]:
                 if not waveform.defined_channels == self.defined_channels:
                     print(f"SequenceWaveform: defined channels {self.defined_channels} do not match {waveform.defined_channels} ")
            raise ValueError(
                "SequenceWaveform cannot be constructed from waveforms of different"
                "defined channels."
            )

    @classmethod
    def from_sequence(cls, waveforms: Sequence['Waveform']) -> 'Waveform':
        """Returns a waveform the represents the given sequence of waveforms. Applies some optimizations."""
        assert waveforms, "Sequence must not be empty"
        if len(waveforms) == 1:
            return waveforms[0]

        flattened = []
        constant_values = waveforms[0].constant_value_dict()
        for wf in waveforms:
            if constant_values and constant_values != wf.constant_value_dict():
                constant_values = None
            if isinstance(wf, cls):
                flattened.extend(wf.sequenced_waveforms)
            else:
                flattened.append(wf)
        if constant_values is None:
            return cls(sub_waveforms=flattened)
        else:
            duration = sum(wf.duration for wf in flattened)
            return ConstantWaveform.from_mapping(duration, constant_values)

    def is_constant(self) -> bool:
        # only correct if from_sequence is used for construction
        return False

    def constant_value_dict(self) -> Optional[Mapping[ChannelID, float]]:
        # only correct if from_sequence is used for construction
        return None

    def constant_value(self, channel: ChannelID) -> Optional[float]:
        v = None
        for wf in self._sequenced_waveforms:
            wf_cv = wf.constant_value(channel)
            if wf_cv is None:
                return None
            elif wf_cv == v:
                continue
            elif v is None:
                v = wf_cv
            else:
                assert v != wf_cv
                return None
        return v

    @property
    def defined_channels(self) -> AbstractSet[ChannelID]:
        return self._sequenced_waveforms[0].defined_channels

    def unsafe_sample(self,
                      channel: ChannelID,
                      sample_times: np.ndarray,
                      output_array: Union[np.ndarray, None] = None) -> np.ndarray:
        if output_array is None:
            output_array = _ALLOCATION_FUNCTION(sample_times, **_ALLOCATION_FUNCTION_KWARGS)
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
        return SequenceWaveform.from_sequence([
            sub_waveform.unsafe_get_subset_for_channels(channels & sub_waveform.defined_channels)
            for sub_waveform in self._sequenced_waveforms if sub_waveform.defined_channels & channels])

    @property
    def sequenced_waveforms(self) -> Sequence[Waveform]:
        return self._sequenced_waveforms

    def __repr__(self):
        return f"{type(self).__name__}({self._sequenced_waveforms})"


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

    __slots__ = ('_sub_waveforms', '_defined_channels')

    def __init__(self, sub_waveforms: List[Waveform]) -> None:
        """Create a new MultiChannelWaveform instance.
        Use `MultiChannelWaveform.from_parallel` for optimal construction.

        Requires a list of subwaveforms in the form (Waveform, List(int)) where the list defines
        the channel mapping, i.e., a value y at index x in the list means that channel x of the
        subwaveform will be mapped to channel y of this MultiChannelWaveform object.

        Args:
            sub_waveforms: The list of sub waveforms of this
                MultiChannelWaveform. List might get sorted!
        Raises:
            ValueError, if a channel mapping is out of bounds of the channels defined by this
                MultiChannelWaveform
            ValueError, if several subwaveform channels are assigned to a single channel of this
                MultiChannelWaveform
            ValueError, if subwaveforms have inconsistent durations
        """

        if not sub_waveforms:
            raise ValueError(
                "MultiChannelWaveform cannot be constructed without channel waveforms."
            )

        # sort the waveforms with their defined channels to make compare key reproducible
        if not isinstance(sub_waveforms, list):
            sub_waveforms = list(sub_waveforms)
        sub_waveforms.sort(key=lambda wf: wf._sort_key_for_channels())

        super().__init__(duration=sub_waveforms[0].duration)
        self._sub_waveforms = tuple(sub_waveforms)

        defined_channels = set()
        for waveform in self._sub_waveforms:
            if waveform.defined_channels & defined_channels:
                raise ValueError('Channel may not be defined in multiple waveforms',
                                 waveform.defined_channels & defined_channels)
            defined_channels |= waveform.defined_channels
        self._defined_channels = frozenset(defined_channels)

        if not all(isclose(waveform.duration, self.duration) for waveform in self._sub_waveforms[1:]):
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

    @staticmethod
    def from_parallel(waveforms: Sequence[Waveform]) -> Waveform:
        assert waveforms, "ARgument must not be empty"
        if len(waveforms) == 1:
            return waveforms[0]

        # we do not look at constant values here because there is no benefit. We would need to construct a new
        # MultiChannelWaveform anyways

        # avoid unnecessary multi channel nesting
        flattened = []
        for waveform in waveforms:
            if isinstance(waveform, MultiChannelWaveform):
                flattened.extend(waveform._sub_waveforms)
            else:
                flattened.append(waveform)

        return MultiChannelWaveform(flattened)

    def is_constant(self) -> bool:
        return all(wf.is_constant() for wf in self._sub_waveforms)

    def constant_value(self, channel: ChannelID) -> Optional[float]:
        return self[channel].constant_value(channel)

    def constant_value_dict(self) -> Optional[Mapping[ChannelID, float]]:
        d = {}
        for wf in self._sub_waveforms:
            wf_d = wf.constant_value_dict()
            if wf_d is None:
                return None
            else:
                d.update(wf_d)
        return d

    @property
    def duration(self) -> TimeType:
        return self._sub_waveforms[0].duration

    def __getitem__(self, key: ChannelID) -> Waveform:
        for waveform in self._sub_waveforms:
            if key in waveform.defined_channels:
                return waveform
        raise KeyError('Unknown channel ID: {}'.format(key), key)

    @property
    def defined_channels(self) -> AbstractSet[ChannelID]:
        return self._defined_channels

    @property
    def compare_key(self) -> Any:
        # sort with channels
        return self._sub_waveforms

    def unsafe_sample(self,
                      channel: ChannelID,
                      sample_times: np.ndarray,
                      output_array: Union[np.ndarray, None] = None) -> np.ndarray:
        return self[channel].unsafe_sample(channel, sample_times, output_array)

    def unsafe_get_subset_for_channels(self, channels: AbstractSet[ChannelID]) -> 'Waveform':
        relevant_sub_waveforms = [swf for swf in self._sub_waveforms if swf.defined_channels & channels]
        if len(relevant_sub_waveforms) == 1:
            return relevant_sub_waveforms[0].get_subset_for_channels(channels)
        elif len(relevant_sub_waveforms) > 1:
            return MultiChannelWaveform.from_parallel(
                [sub_waveform.get_subset_for_channels(channels & sub_waveform.defined_channels)
                 for sub_waveform in relevant_sub_waveforms])
        else:
            raise KeyError('Unknown channels: {}'.format(channels))

    def __repr__(self):
        return f"{type(self).__name__}({self._sub_waveforms!r})"


class RepetitionWaveform(Waveform):
    """This class allows putting multiple PulseTemplate together in one waveform on the hardware."""

    __slots__ = ('_body', '_repetition_count')

    def __init__(self, body: Waveform, repetition_count: int):
        repetition_count = checked_int_cast(repetition_count)
        if repetition_count < 1 or not isinstance(repetition_count, int):
            raise ValueError('Repetition count must be an integer >0')

        super().__init__(duration=body.duration * repetition_count)
        self._body = body
        self._repetition_count = repetition_count

    @classmethod
    def from_repetition_count(cls, body: Waveform, repetition_count: int) -> Waveform:
        constant_values = body.constant_value_dict()
        if constant_values is None:
            return RepetitionWaveform(body, repetition_count)
        else:
            return ConstantWaveform.from_mapping(body.duration * repetition_count, constant_values)

    @property
    def defined_channels(self) -> AbstractSet[ChannelID]:
        return self._body.defined_channels

    def unsafe_sample(self,
                      channel: ChannelID,
                      sample_times: np.ndarray,
                      output_array: Union[np.ndarray, None] = None) -> np.ndarray:
        if output_array is None:
            output_array = _ALLOCATION_FUNCTION(sample_times, **_ALLOCATION_FUNCTION_KWARGS)
        body_duration = self._body.duration
        time = 0
        for _ in range(self._repetition_count):
            end = time + body_duration
            indices = slice(*np.searchsorted(sample_times, (float(time), float(end)), 'left'))
            self._body.unsafe_sample(channel=channel,
                                     sample_times=sample_times[indices] - float(time),
                                     output_array=output_array[indices])
            time = end
        return output_array

    @property
    def compare_key(self) -> Tuple[Any, int]:
        return self._body.compare_key, self._repetition_count

    def unsafe_get_subset_for_channels(self, channels: AbstractSet[ChannelID]) -> Waveform:
        return RepetitionWaveform.from_repetition_count(
            body=self._body.unsafe_get_subset_for_channels(channels),
            repetition_count=self._repetition_count)

    def is_constant(self) -> bool:
        return self._body.is_constant()

    def constant_value(self, channel: ChannelID) -> Optional[float]:
        return self._body.constant_value(channel)

    def constant_value_dict(self) -> Optional[Mapping[ChannelID, float]]:
        return self._body.constant_value_dict()

    def __repr__(self):
        return f"{type(self).__name__}(body={self._body!r}, repetition_count={self._repetition_count!r})"


class TransformingWaveform(Waveform):
    __slots__ = ('_inner_waveform', '_transformation', '_cached_data', '_cached_times')

    def __init__(self, inner_waveform: Waveform, transformation: Transformation):
        """"""
        super(TransformingWaveform, self).__init__(duration=inner_waveform.duration)
        self._inner_waveform = inner_waveform
        self._transformation = transformation

        # cache data of inner channels based identified and invalidated by the sample times
        self._cached_data = None
        self._cached_times = lambda: None

    @classmethod
    def from_transformation(cls, inner_waveform: Waveform, transformation: Transformation) -> Waveform:
        constant_values = inner_waveform.constant_value_dict()

        if constant_values is None or not transformation.is_constant_invariant():
            return cls(inner_waveform, transformation)

        transformed_constant_values = {key: float(value) for key, value in transformation(0., constant_values).items()}
        return ConstantWaveform.from_mapping(inner_waveform.duration, transformed_constant_values)

    def is_constant(self) -> bool:
        # only true if `from_transformation` was used
        return False

    def constant_value_dict(self) -> Optional[Mapping[ChannelID, float]]:
        # only true if `from_transformation` was used
        return None

    def constant_value(self, channel: ChannelID) -> Optional[float]:
        if not self._transformation.is_constant_invariant():
            return None
        in_channels = self._transformation.get_input_channels({channel})
        in_values = {ch: self._inner_waveform.constant_value(ch) for ch in in_channels}
        if any(val is None for val in in_values.values()):
            return None
        else:
            return self._transformation(0., in_values)[channel]

    @property
    def inner_waveform(self) -> Waveform:
        return self._inner_waveform

    @property
    def transformation(self) -> Transformation:
        return self._transformation

    @property
    def defined_channels(self) -> AbstractSet[ChannelID]:
        return self.transformation.get_output_channels(self.inner_waveform.defined_channels)

    @property
    def compare_key(self) -> Tuple[Waveform, Transformation]:
        return self.inner_waveform, self.transformation

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
    __slots__ = ('_inner_waveform', '_channel_subset')

    def __init__(self, inner_waveform: Waveform, channel_subset: Set[ChannelID]):
        super().__init__(duration=inner_waveform.duration)
        self._inner_waveform = inner_waveform
        self._channel_subset = frozenset(channel_subset)

    @property
    def inner_waveform(self) -> Waveform:
        return self._inner_waveform

    @property
    def defined_channels(self) -> FrozenSet[ChannelID]:
        return self._channel_subset

    @property
    def compare_key(self) -> Tuple[frozenset, Waveform]:
        return self.defined_channels, self.inner_waveform

    def unsafe_get_subset_for_channels(self, channels: Set[ChannelID]) -> Waveform:
        return self.inner_waveform.get_subset_for_channels(channels)

    def unsafe_sample(self,
                      channel: ChannelID,
                      sample_times: np.ndarray,
                      output_array: Union[np.ndarray, None] = None) -> np.ndarray:
        return self.inner_waveform.unsafe_sample(channel, sample_times, output_array)

    def constant_value_dict(self) -> Optional[Mapping[ChannelID, float]]:
        d = self._inner_waveform.constant_value_dict()
        if d is not None:
            return {ch: d[ch] for ch in self._channel_subset}

    def constant_value(self, channel: ChannelID) -> Optional[float]:
        if channel not in self._channel_subset:
            raise KeyError(channel)
        return self._inner_waveform.constant_value(channel)


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

    __slots__ = ('_lhs', '_rhs', '_arithmetic_operator')

    def __init__(self,
                 lhs: Waveform,
                 arithmetic_operator: str,
                 rhs: Waveform):
        super().__init__(duration=lhs.duration)
        self._lhs = lhs
        self._rhs = rhs
        self._arithmetic_operator = arithmetic_operator

        assert np.isclose(float(self._lhs.duration), float(self._rhs.duration))
        assert arithmetic_operator in self.operator_map

    @classmethod
    def from_operator(cls, lhs: Waveform, arithmetic_operator: str, rhs: Waveform):
        # one could optimize rhs_cv to being only created if lhs_cv is not None but this makes the code harder to read
        lhs_cv = lhs.constant_value_dict()
        rhs_cv = rhs.constant_value_dict()
        if lhs_cv is None or rhs_cv is None:
            return cls(lhs, arithmetic_operator, rhs)

        else:
            constant_values = dict(lhs_cv)
            op = cls.operator_map[arithmetic_operator]
            rhs_op = cls.rhs_only_map[arithmetic_operator]

            for ch, rhs_val in rhs_cv.items():
                if ch in constant_values:
                    constant_values[ch] = op(constant_values[ch], rhs_val)
                else:
                    constant_values[ch] = rhs_op(rhs_val)

            duration = lhs.duration
            assert isclose(duration, rhs.duration)

            return ConstantWaveform.from_mapping(duration, constant_values)

    def constant_value(self, channel: ChannelID) -> Optional[float]:
        if channel not in self._rhs.defined_channels:
            return self._lhs.constant_value(channel)
        rhs = self._rhs.constant_value(channel)
        if rhs is None:
            return None

        if channel in self._lhs.defined_channels:
            lhs = self._lhs.constant_value(channel)
            if lhs is None:
                return None

            return self.operator_map[self._arithmetic_operator](lhs, rhs)
        else:
            return self.rhs_only_map[self._arithmetic_operator](rhs)

    def is_constant(self) -> bool:
        # only correct if from_operator is used
        return False

    def constant_value_dict(self) -> Optional[Mapping[ChannelID, float]]:
        # only correct if from_operator is used
        return None

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
    def defined_channels(self) -> AbstractSet[ChannelID]:
        return self._lhs.defined_channels | self._rhs.defined_channels

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
    # TODO: Use Protocol to enforce that it accepts second argument has the keyword out
    Functor = callable

    __slots__ = ('_inner_waveform', '_functor')

    """Apply a channel wise functor that works inplace to all results. The functor must accept two arguments"""
    def __init__(self, inner_waveform: Waveform, functor: Mapping[ChannelID, Functor]):
        super(FunctorWaveform, self).__init__(duration=inner_waveform.duration)
        self._inner_waveform = inner_waveform
        self._functor = dict(functor.items())

        assert set(functor.keys()) == inner_waveform.defined_channels, ("There is no default identity mapping (yet)."
                                                                        "File an issue on github if you need it.")

    @classmethod
    def from_functor(cls, inner_waveform: Waveform, functor: Mapping[ChannelID, Functor]):
        constant_values = inner_waveform.constant_value_dict()
        if constant_values is None:
            return FunctorWaveform(inner_waveform, functor)

        funced_constant_values = {ch: functor[ch](val) for ch, val in constant_values.items()}
        return ConstantWaveform.from_mapping(inner_waveform.duration, funced_constant_values)

    def is_constant(self) -> bool:
        # only correct if `from_functor` was used
        return False

    def constant_value_dict(self) -> Optional[Mapping[ChannelID, float]]:
        # only correct if `from_functor` was used
        return None

    def constant_value(self, channel: ChannelID) -> Optional[float]:
        inner = self._inner_waveform.constant_value(channel)
        if inner is None:
            return None
        else:
            return self._functor[channel](inner)

    @property
    def defined_channels(self) -> AbstractSet[ChannelID]:
        return self._inner_waveform.defined_channels

    def unsafe_sample(self,
                      channel: ChannelID,
                      sample_times: np.ndarray,
                      output_array: Union[np.ndarray, None] = None) -> np.ndarray:
        inner_output = self._inner_waveform.unsafe_sample(channel, sample_times, output_array)
        return self._functor[channel](inner_output, out=inner_output)

    def unsafe_get_subset_for_channels(self, channels: Set[ChannelID]) -> Waveform:
        return FunctorWaveform.from_functor(
            self._inner_waveform.unsafe_get_subset_for_channels(channels),
            {ch: self._functor[ch] for ch in channels})

    @property
    def compare_key(self) -> Tuple[Waveform, FrozenSet]:
        return self._inner_waveform, frozenset(self._functor.items())


class ReversedWaveform(Waveform):
    """Reverses the inner waveform in time."""

    __slots__ = ('_inner',)

    def __init__(self, inner: Waveform):
        super().__init__(duration=inner.duration)
        self._inner = inner

    @classmethod
    def from_to_reverse(cls, inner: Waveform) -> Waveform:
        if inner.constant_value_dict():
            return inner
        else:
            return cls(inner)

    def unsafe_sample(self, channel: ChannelID, sample_times: np.ndarray,
                      output_array: Union[np.ndarray, None] = None) -> np.ndarray:
        inner_sample_times = (float(self.duration) - sample_times)[::-1]
        if output_array is None:
            return self._inner.unsafe_sample(channel, inner_sample_times, None)[::-1]
        else:
            inner_output_array = output_array[::-1]
            inner_output_array = self._inner.unsafe_sample(channel, inner_sample_times, output_array=inner_output_array)
            if inner_output_array.base not in (output_array, output_array.base):
                # TODO: is there a guarantee by numpy we never end up here?
                output_array[:] = inner_output_array[::-1]
            return output_array

    @property
    def defined_channels(self) -> AbstractSet[ChannelID]:
        return self._inner.defined_channels

    def unsafe_get_subset_for_channels(self, channels: AbstractSet[ChannelID]) -> 'Waveform':
        return ReversedWaveform.from_to_reverse(self._inner.unsafe_get_subset_for_channels(channels))

    @property
    def compare_key(self) -> Hashable:
        return self._inner.compare_key

    def reversed(self) -> 'Waveform':
        return self._inner
