from typing import Mapping, Set, Dict
from abc import abstractmethod

import numpy as np
import pandas as pd

from qctoolkit import ChannelID
from qctoolkit.comparable import Comparable


class Transformation(Comparable):
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

        return self._matrix @ data_in

    def get_output_channels(self, input_channels: Set[ChannelID]) -> Set[ChannelID]:
        if input_channels != set(self._matrix.columns):
            raise KeyError('Invalid input channels', input_channels, set(self._matrix.columns))

        return set(self._matrix.index)

    @property
    def compare_key(self) -> Dict[ChannelID, Dict[ChannelID, float]]:
        return self._matrix.to_dict()
