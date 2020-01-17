from abc import ABC, abstractmethod
from typing import Iterable, Optional

from .base_features import BaseFeature, FeatureAble


class BaseAWGFeature(BaseFeature, ABC):
    """Base class for features that are used for `AWGDevice`s"""
    pass


class BaseAWGChannelFeature(BaseFeature, ABC):
    """Base class for features that are used for `AWGChannel`s"""
    pass


class BaseAWGChannelTupleFeature(BaseFeature, ABC):
    """Base class for features that are used for `AWGChannelTuple`s"""
    pass


class BaseAWG(FeatureAble[BaseAWGFeature], ABC):
    """Base class for all drivers of all arbitrary waveform generators"""

    def __init__(self, name: str):
        """
        Args:
            name: The name of the device as a String
        """
        super().__init__()
        self._name = name

    def __del__(self):
        self.cleanup()

    @property
    def name(self) -> str:
        """Returns the name of a Device as a String"""
        return self._name

    @abstractmethod
    def cleanup(self) -> None:
        """Function for cleaning up the dependencies of the device"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def channels(self) -> Iterable["BaseAWGChannel"]:
        """Returns a list of all channels of a Device"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def channel_tuples(self) -> Iterable["BaseAWGChannelTuple"]:
        """Returns a list of all channel tuples of a list"""
        raise NotImplementedError()


class BaseAWGChannelTuple(FeatureAble[BaseAWGChannelTupleFeature], ABC):
    """Base class for all groups of synchronized channels of an AWG"""

    def __init__(self, idn: int):
        """
        Args:
             idn: The identification number of a channel tuple
        """
        super().__init__()

        self._idn = idn

    @property
    def idn(self) -> int:
        """Returns the identification number of a channel tuple"""
        return self._idn

    @property
    @abstractmethod
    def sample_rate(self) -> float:
        """Returns the sample rate of a channel tuple as a float"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def device(self) -> BaseAWG:
        """Returns the device which the channel tuple belong to"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def channels(self) -> Iterable["BaseAWGChannel"]:
        """Returns a list of all channels of the channel tuple"""
        raise NotImplementedError()


class BaseAWGChannel(FeatureAble[BaseAWGChannelFeature], ABC):
    """Base class for a single channel of an AWG"""

    def __init__(self, idn: int):
        """
        Args:
            idn: The identification number of a channel
        """
        super().__init__()
        self._idn = idn

    @property
    def idn(self) -> int:
        """Returns the identification number of a channel"""
        return self._idn

    @property
    @abstractmethod
    def device(self) -> BaseAWG:
        """Returns the device which the channel belongs to"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def channel_tuple(self) -> Optional[BaseAWGChannelTuple]:
        """Returns the channel tuple which a channel belongs to"""
        raise NotImplementedError()

    @abstractmethod
    def _set_channel_tuple(self, channel_tuple) -> None:
        """
        Sets the channel tuple which a channel belongs to

        Args:
            channel_tuple: reference to the channel tuple
        """
        raise NotImplementedError()
