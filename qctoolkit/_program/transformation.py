from typing import Mapping, Set, Dict, Tuple
from abc import abstractmethod

import numpy as np
import pandas as pd

from qctoolkit import ChannelID
from qctoolkit.comparable import Comparable
from qctoolkit.utils.types import SingletonABCMeta


class Transformation(Comparable):
    _identity_singleton = None
    """Transforms numeric time-voltage values for multiple channels to other time-voltage values. The number and names
     of input and output channels might differ."""

    @abstractmethod
    def __call__(self, time: np.ndarray, data: pd.DataFrame) -> pd.DataFrame:
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

    def chain(self, next_transformation: 'Transformation') -> 'Transformation':
        if next_transformation is IdentityTransformation():
            return self
        else:
            return chain_transformations(self, next_transformation)


class IdentityTransformation(Transformation, metaclass=SingletonABCMeta):
    def __call__(self, time: np.ndarray, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def get_output_channels(self, input_channels: Set[ChannelID]) -> Set[ChannelID]:
        return input_channels

    @property
    def compare_key(self) -> None:
        return None

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

    def __call__(self, time: np.ndarray, data: pd.DataFrame) -> pd.DataFrame:
        for transformation in self._transformations:
            data = transformation(time, data)
        return data

    @property
    def compare_key(self) -> Tuple[Transformation, ...]:
        return self._transformations

    def chain(self, next_transformation) -> 'ChainedTransformation':
        return chain_transformations(*self.transformations, next_transformation)


class LinearTransformation(Transformation):
    def __init__(self, transformation_matrix: pd.DataFrame):
        """

        Args:
            transformation_matrix: columns are input and index are output channels
        """
        self._matrix = transformation_matrix

    def __call__(self, time: np.ndarray, data: pd.DataFrame) -> Mapping[ChannelID, np.ndarray]:
        data_in = pd.DataFrame(data)
        if set(data_in.index) != set(self._matrix.columns):
            raise KeyError('Invalid input channels', set(data_in.index), set(self._matrix.columns))

        return self._matrix.dot(data_in)

    def get_output_channels(self, input_channels: Set[ChannelID]) -> Set[ChannelID]:
        if input_channels != set(self._matrix.columns):
            raise KeyError('Invalid input channels', input_channels, set(self._matrix.columns))

        return set(self._matrix.index)

    @property
    def compare_key(self) -> Dict[ChannelID, Dict[ChannelID, float]]:
        return frozenset((key, frozenset(value.items()))
                         for key, value in self._matrix.to_dict().items())


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