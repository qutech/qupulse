from abc import ABC, abstractmethod
from typing import Collection, Optional

from qupulse.hardware.awgs.base_features import Feature, FeatureAble


__all__ = ["AWGDevice", "AWGChannelTuple", "AWGChannel", "AWGMarkerChannel", "AWGDeviceFeature", "AWGChannelFeature",
           "AWGChannelTupleFeature"]


class AWGDeviceFeature(Feature, ABC):
    """Base class for features that are used for `AWGDevice`s"""
    def __init__(self):
        super().__init__(AWGDevice)


class AWGChannelFeature(Feature, ABC):
    """Base class for features that are used for `AWGChannel`s"""
    def __init__(self):
        super().__init__(AWGChannel)


class AWGChannelTupleFeature(Feature, ABC):
    """Base class for features that are used for `AWGChannelTuple`s"""
    def __init__(self):
        super().__init__(AWGChannelTuple)


class AWGDevice(FeatureAble[AWGDeviceFeature], ABC):
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
    def channels(self) -> Collection["AWGChannel"]:
        """Returns a list of all channels of a Device"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def marker_channels(self) -> Collection["AWGMarkerChannel"]:
        """Returns a list of all marker channels of a device. The collection may be empty"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def channel_tuples(self) -> Collection["AWGChannelTuple"]:
        """Returns a list of all channel tuples of a list"""
        raise NotImplementedError()


class AWGChannelTuple(FeatureAble[AWGChannelTupleFeature], ABC):
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
    def name(self) -> str:
        """Returns the name of a channel tuple"""
        return "{dev}_CT{idn}".format(dev=self.device.name, idn=self.idn)

    @property
    @abstractmethod
    def sample_rate(self) -> float:
        """Returns the sample rate of a channel tuple as a float"""
        raise NotImplementedError()

    # Optional sample_rate-setter
    # @sample_rate.setter
    # def sample_rate(self, sample_rate: float) -> None:

    @property
    @abstractmethod
    def device(self) -> AWGDevice:
        """Returns the device which the channel tuple belong to"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def channels(self) -> Collection["AWGChannel"]:
        """Returns a list of all channels of the channel tuple"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def marker_channels(self) -> Collection["AWGMarkerChannel"]:
        """Returns a list of all marker channels of the channel tuple. The collection may be empty"""
        raise NotImplementedError()


class _BaseAWGChannel(ABC):
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
    def device(self) -> AWGDevice:
        """Returns the device which the channel belongs to"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def channel_tuple(self) -> Optional[AWGChannelTuple]:
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


class AWGChannel(_BaseAWGChannel, FeatureAble[AWGChannelFeature], ABC):
    """Base class for a single channel of an AWG"""
    @property
    def name(self) -> str:
        """Returns the name of a channel"""
        return "{dev}_C{idn}".format(dev=self.device.name, idn=self.idn)
    
    
class AWGMarkerChannel(_BaseAWGChannel, ABC):
    """Base class for a single marker channel of an AWG"""
    @property
    def name(self) -> str:
        """Returns the name of a marker channel"""
        return "{dev}_M{idn}".format(dev=self.device.name, idn=self.idn)
