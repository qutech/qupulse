from typing import Mapping, Set, Tuple, Sequence
from abc import abstractmethod

import numpy as np

from qctoolkit import ChannelID
from qctoolkit.comparable import Comparable
from qctoolkit.utils.types import SingletonABCMeta


class Transformation(Comparable):
    _identity_singleton = None
    """Transforms numeric time-voltage values for multiple channels to other time-voltage values. The number and names
     of input and output channels might differ."""

    @abstractmethod
    def __call__(self, time: np.ndarray, data: Mapping[ChannelID, np.ndarray]) -> Mapping[ChannelID, np.ndarray]:
        """Apply transformation to data
        Args:
            time:
            data:

        Returns:
            transformed: A DataFrame that has been transformed with index == output_channels
        """

    @abstractmethod
    def get_output_channels(self, input_channels: Set[ChannelID]) -> Set[ChannelID]:
        """Return the channel identifiers"""

    @abstractmethod
    def get_input_channels(self, output_channels: Set[ChannelID]) -> Set[ChannelID]:
        """Channels that are required for getting data for the requested output channel"""

    def chain(self, next_transformation: 'Transformation') -> 'Transformation':
        if next_transformation is IdentityTransformation():
            return self
        else:
            return chain_transformations(self, next_transformation)


class IdentityTransformation(Transformation, metaclass=SingletonABCMeta):
    def __call__(self, time: np.ndarray, data: Mapping[ChannelID, np.ndarray]) -> Mapping[ChannelID, np.ndarray]:
        return data

    def get_output_channels(self, input_channels: Set[ChannelID]) -> Set[ChannelID]:
        return input_channels

    @property
    def compare_key(self) -> None:
        return None

    def get_input_channels(self, output_channels: Set[ChannelID]) -> Set[ChannelID]:
        return output_channels

    def chain(self, next_transformation: Transformation) -> Transformation:
        return next_transformation


class ChainedTransformation(Transformation):
    def __init__(self, *transformations: Transformation):
        self._transformations = transformations

    @property
    def transformations(self) -> Tuple[Transformation, ...]:
        return self._transformations

    def get_output_channels(self, input_channels: Set[ChannelID]) -> Set[ChannelID]:
        for transformation in self._transformations:
            input_channels = transformation.get_output_channels(input_channels)
        return input_channels

    def get_input_channels(self, output_channels: Set[ChannelID]) -> Set[ChannelID]:
        for transformation in reversed(self._transformations):
            output_channels = transformation.get_input_channels(output_channels)
        return output_channels

    def __call__(self, time: np.ndarray, data: Mapping[ChannelID, np.ndarray]) -> Mapping[ChannelID, np.ndarray]:
        for transformation in self._transformations:
            data = transformation(time, data)
        return data

    @property
    def compare_key(self) -> Tuple[Transformation, ...]:
        return self._transformations

    def chain(self, next_transformation) -> 'ChainedTransformation':
        return chain_transformations(*self.transformations, next_transformation)


class LinearTransformation(Transformation):
    def __init__(self,
                 transformation_matrix: np.ndarray,
                 input_channels: Sequence[ChannelID],
                 output_channels: Sequence[ChannelID]):
        """

        Args:
            transformation_matrix: columns are input and index are output channels
        """
        assert transformation_matrix.shape == (len(output_channels), len(input_channels))

        output_sorter = np.argsort(output_channels)
        transformation_matrix = transformation_matrix[output_sorter, :]

        input_sorter = np.argsort(input_channels)
        transformation_matrix = transformation_matrix[:, input_sorter]

        self._matrix = transformation_matrix
        self._input_channels = tuple(sorted(input_channels))
        self._output_channels = tuple(sorted(output_channels))

    @classmethod
    def from_pandas(cls, transformation: 'pandas.DataFrame') -> 'LinearTransformation':
        return cls(transformation.values, transformation.columns, transformation.index)

    def __call__(self, time: np.ndarray, data: Mapping[ChannelID, np.ndarray]) -> Mapping[ChannelID, np.ndarray]:
        try:
            data_in = np.stack(data[in_channel] for in_channel in self._input_channels)
        except KeyError as error:
            raise KeyError('Invalid input channels', set(data.keys()), set(self._input_channels)) from error

        data_out = {forwarded_channel: data[forwarded_channel]
                    for forwarded_channel in set(data.keys()).difference(self._input_channels)}

        transformed_data = self._matrix @ data_in

        for idx, out_channel in enumerate(self._output_channels):
            data_out[out_channel] = transformed_data[idx, :]

        return data_out

    def get_output_channels(self, input_channels: Set[ChannelID]) -> Set[ChannelID]:
        if not input_channels.issuperset(self._input_channels):
            raise KeyError('Invalid input channels', input_channels, set(self._input_channels))

        return input_channels.difference(self._input_channels).union(self._output_channels)

    def get_input_channels(self, output_channels: Set[ChannelID]) -> Set[ChannelID]:
        forwarded = output_channels.difference(self._output_channels)
        if not forwarded.isdisjoint(self._input_channels):
            raise KeyError('Is input channel', forwarded.intersection(self._input_channels))
        elif output_channels.isdisjoint(self._output_channels):
            return output_channels
        else:
            return forwarded.union(self._input_channels)

    @property
    def compare_key(self) -> Tuple[Tuple[ChannelID], Tuple[ChannelID], bytes]:
        return self._input_channels, self._output_channels, self._matrix.tobytes()


def chain_transformations(*transformations: Transformation) -> Transformation:
    parsed_transformations = []
    for transformation in transformations:
        if transformation is IdentityTransformation():
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