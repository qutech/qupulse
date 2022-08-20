from typing import Any, Mapping, Set, Tuple, Sequence, AbstractSet, Union, TYPE_CHECKING
from abc import abstractmethod
from numbers import Real

import numpy as np

from qupulse import ChannelID
from qupulse.comparable import Comparable
from qupulse.utils.types import SingletonABCMeta


class Transformation(Comparable):
    _identity_singleton = None
    """Transforms numeric time-voltage values for multiple channels to other time-voltage values. The number and names
     of input and output channels might differ."""

    @abstractmethod
    def __call__(self, time: Union[np.ndarray, float],
                 data: Mapping[ChannelID, Union[np.ndarray, float]]) -> Mapping[ChannelID, Union[np.ndarray, float]]:
        """Apply transformation to data
        Args:
            time:
            data:

        Returns:
            transformed: A DataFrame that has been transformed with index == output_channels
        """

    @abstractmethod
    def get_output_channels(self, input_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        """Return the channel identifiers"""

    @abstractmethod
    def get_input_channels(self, output_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        """Channels that are required for getting data for the requested output channel"""

    def chain(self, next_transformation: 'Transformation') -> 'Transformation':
        if next_transformation is IdentityTransformation():
            return self
        else:
            return chain_transformations(self, next_transformation)

    def is_constant_invariant(self):
        """Signals if the transformation always maps constants to constants."""
        return False


class IdentityTransformation(Transformation, metaclass=SingletonABCMeta):
    def __call__(self, time: Union[np.ndarray, float],
                 data: Mapping[ChannelID, Union[np.ndarray, float]]) -> Mapping[ChannelID, Union[np.ndarray, float]]:
        return data

    def get_output_channels(self, input_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        return input_channels

    @property
    def compare_key(self) -> None:
        return None

    def get_input_channels(self, output_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        return output_channels

    def chain(self, next_transformation: Transformation) -> Transformation:
        return next_transformation

    def __repr__(self):
        return 'IdentityTransformation()'

    def is_constant_invariant(self):
        """Signals if the transformation always maps constants to constants."""
        return True


class ChainedTransformation(Transformation):
    def __init__(self, *transformations: Transformation):
        self._transformations = transformations

    @property
    def transformations(self) -> Tuple[Transformation, ...]:
        return self._transformations

    def get_output_channels(self, input_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        for transformation in self._transformations:
            input_channels = transformation.get_output_channels(input_channels)
        return input_channels

    def get_input_channels(self, output_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        for transformation in reversed(self._transformations):
            output_channels = transformation.get_input_channels(output_channels)
        return output_channels

    def __call__(self, time: Union[np.ndarray, float],
                 data: Mapping[ChannelID, Union[np.ndarray, float]]) -> Mapping[ChannelID, Union[np.ndarray, float]]:
        for transformation in self._transformations:
            data = transformation(time, data)
        return data

    @property
    def compare_key(self) -> Tuple[Transformation, ...]:
        return self._transformations

    def chain(self, next_transformation) -> Transformation:
        return chain_transformations(*self.transformations, next_transformation)

    def __repr__(self):
        return 'ChainedTransformation%r' % (self._transformations,)

    def is_constant_invariant(self):
        """Signals if the transformation always maps constants to constants."""
        return all(trafo.is_constant_invariant() for trafo in self._transformations)


class LinearTransformation(Transformation):
    def __init__(self,
                 transformation_matrix: np.ndarray,
                 input_channels: Sequence[ChannelID],
                 output_channels: Sequence[ChannelID]):
        """

        Args:
            transformation_matrix: Matrix describing the transformation with shape (output_channels, input_channels)
            input_channels: Channel ids of the columns
            output_channels: Channel ids of the rows
        """
        transformation_matrix = np.asarray(transformation_matrix)

        if transformation_matrix.shape != (len(output_channels), len(input_channels)):
            raise ValueError('Shape of transformation matrix does not match to the given channels')

        output_sorter = np.argsort(output_channels)
        transformation_matrix = transformation_matrix[output_sorter, :]

        input_sorter = np.argsort(input_channels)
        transformation_matrix = transformation_matrix[:, input_sorter]

        self._matrix = transformation_matrix
        self._input_channels = tuple(sorted(input_channels))
        self._output_channels = tuple(sorted(output_channels))
        self._input_channels_set = frozenset(self._input_channels)
        self._output_channels_set = frozenset(self._output_channels)

    def __call__(self, time: Union[np.ndarray, float],
                 data: Mapping[ChannelID, Union[np.ndarray, float]]) -> Mapping[ChannelID, Union[np.ndarray, float]]:
        data_out = {forwarded_channel: data[forwarded_channel]
                    for forwarded_channel in set(data.keys()).difference(self._input_channels)}

        if len(data_out) == len(data):
            # only forwarded data
            return data_out

        try:
            data_in = np.stack([data[in_channel] for in_channel in self._input_channels])
        except KeyError as error:
            raise KeyError('Invalid input channels', set(data.keys()), set(self._input_channels)) from error

        transformed_data = self._matrix @ data_in

        for idx, out_channel in enumerate(self._output_channels):
            data_out[out_channel] = transformed_data[idx, ...]

        return data_out

    def get_output_channels(self, input_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        if not input_channels >= self._input_channels_set:
            # input_channels is not a superset of the required input channels
            raise KeyError('Invalid input channels', input_channels, self._input_channels_set)

        return (input_channels - self._input_channels_set) | self._output_channels_set

    def get_input_channels(self, output_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        forwarded = output_channels - self._output_channels_set
        if not forwarded.isdisjoint(self._input_channels):
            raise KeyError('Is input channel', forwarded & self._input_channels_set)
        elif output_channels.isdisjoint(self._output_channels):
            return output_channels
        else:
            return forwarded | self._input_channels_set

    @property
    def compare_key(self) -> Tuple[Tuple[ChannelID], Tuple[ChannelID], bytes]:
        return self._input_channels, self._output_channels, self._matrix.tobytes()

    def __repr__(self):
        return ('LinearTransformation('
                'transformation_matrix={transformation_matrix},'
                'input_channels={input_channels},'
                'output_channels={output_channels})').format(transformation_matrix=self._matrix.tolist(),
                                                             input_channels=self._input_channels,
                                                             output_channels=self._output_channels)

    def is_constant_invariant(self):
        """Signals if the transformation always maps constants to constants."""
        return True


class OffsetTransformation(Transformation):
    def __init__(self, offsets: Mapping[ChannelID, Real]):
        """Adds an offset to each channel specified in offsets.

        Channels not in offsets are forewarded

        Args:
            offsets: Channel -> offset mapping
        """
        self._offsets = dict(offsets.items())

    def __call__(self, time: Union[np.ndarray, float],
                 data: Mapping[ChannelID, Union[np.ndarray, float]]) -> Mapping[ChannelID, Union[np.ndarray, float]]:
        return {channel: channel_values + self._offsets[channel] if channel in self._offsets else channel_values
                for channel, channel_values in data.items()}

    def get_input_channels(self, output_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        return output_channels

    def get_output_channels(self, input_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        return input_channels

    @property
    def compare_key(self) -> frozenset:
        return frozenset(self._offsets.items())

    def __repr__(self):
        return 'OffsetTransformation(%r)' % self._offsets

    def is_constant_invariant(self):
        """Signals if the transformation always maps constants to constants."""
        return True


class ScalingTransformation(Transformation):
    def __init__(self, factors: Mapping[ChannelID, Real]):
        self._factors = dict(factors.items())

    def __call__(self, time: Union[np.ndarray, float],
                 data: Mapping[ChannelID, Union[np.ndarray, float]]) -> Mapping[ChannelID, Union[np.ndarray, float]]:
        return {channel: channel_values * self._factors[channel] if channel in self._factors else channel_values
                for channel, channel_values in data.items()}

    def get_input_channels(self, output_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        return output_channels

    def get_output_channels(self, input_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        return input_channels

    @property
    def compare_key(self) -> frozenset:
        return frozenset(self._factors.items())

    def __repr__(self):
        return 'ScalingTransformation(%r)' % self._factors

    def is_constant_invariant(self):
        """Signals if the transformation always maps constants to constants."""
        return True


try:
    if TYPE_CHECKING:
        import pandas
        PandasDataFrameType = pandas.DataFrame
    else:
        PandasDataFrameType = Any

    def linear_transformation_from_pandas(transformation: PandasDataFrameType) -> LinearTransformation:
        """ Creates a LinearTransformation object out of a pandas data frame.

        Args:
            transformation (pandas.DataFrame): The pandas.DataFrame object out of which a LinearTransformation will be formed.

        Returns:
            the created LinearTransformation instance
        """
        return LinearTransformation(transformation.values, transformation.columns, transformation.index)

    LinearTransformation.from_pandas = linear_transformation_from_pandas
except ImportError:
    pass


class ParallelConstantChannelTransformation(Transformation):
    def __init__(self, channels: Mapping[ChannelID, Real]):
        """Set channel values to given values regardless their former existence

        Args:
            channels: Channels present in this map are set to the given value.
        """
        self._channels = {channel: float(value)
                          for channel, value in channels.items()}

    def __call__(self, time: Union[np.ndarray, float],
                 data: Mapping[ChannelID, Union[np.ndarray, float]]) -> Mapping[ChannelID, Union[np.ndarray, float]]:
        overwritten = {channel: np.full_like(time, fill_value=value, dtype=float)
                       for channel, value in self._channels.items()}
        return {**data, **overwritten}

    @property
    def compare_key(self) -> Tuple[Tuple[ChannelID, float], ...]:
        return tuple(sorted(self._channels.items()))

    def get_input_channels(self, output_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        return output_channels - self._channels.keys()

    def get_output_channels(self, input_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        return input_channels | self._channels.keys()

    def __repr__(self):
        return 'ParallelConstantChannelTransformation(%r)' % self._channels

    def is_constant_invariant(self):
        """Signals if the transformation always maps constants to constants."""
        return True


def chain_transformations(*transformations: Transformation) -> Transformation:
    parsed_transformations = []
    for transformation in transformations:
        if transformation is IdentityTransformation() or transformation is None:
            pass
        elif isinstance(transformation, ChainedTransformation):
            parsed_transformations.extend(transformation.transformations)
        else:
            parsed_transformations.append(transformation)
    if len(parsed_transformations) == 0:
        return IdentityTransformation()
    elif len(parsed_transformations) == 1:
        return parsed_transformations[0]
    else:
        return ChainedTransformation(*parsed_transformations)