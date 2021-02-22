from abc import ABC, abstractmethod
from typing import Optional, Tuple, Sequence, Callable, List
from collections import OrderedDict

import numpy

from qupulse._program._loop import Loop
from qupulse._program.waveforms import Waveform
from qupulse.hardware.feature_awg import channel_tuple_wrapper
from qupulse.hardware.feature_awg.base_features import Feature, FeatureAble
from qupulse.hardware.util import get_sample_times
from qupulse.utils.types import Collection, TimeType, ChannelID

__all__ = ["AWGDevice", "AWGChannelTuple", "AWGChannel", "AWGMarkerChannel", "AWGDeviceFeature", "AWGChannelFeature",
           "AWGChannelTupleFeature"]


class AWGDeviceFeature(Feature, ABC):
    """Base class for features that are used for `AWGDevice`s"""
    def __init__(self):
        super().__init__(AWGDevice)


class AWGChannelFeature(Feature, ABC):
    """Base class for features that are used for `AWGChannel`s"""
    def __init__(self):
        super().__init__(_BaseAWGChannel)


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

    #def __del__(self):
    #    self.cleanup()

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
    @abstractmethod
    def channel_tuple_adapter(self) -> channel_tuple_wrapper:
        pass

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


class _BaseAWGChannel(FeatureAble[AWGChannelFeature], ABC):
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


class AWGChannel(_BaseAWGChannel, ABC):
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


class ProgramOverwriteException(Exception):

    def __init__(self, name) -> None:
        super().__init__()
        self.name = name

    def __str__(self) -> str:
        return "A program with the given name '{}' is already present on the device." \
               " Use force to overwrite.".format(self.name)


class ProgramEntry:
    """This is a helper class for implementing awgs drivers. A driver can subclass it to help organizing sampled
    waveforms"""
    def __init__(self, loop: Loop,
                 channels: Tuple[Optional[ChannelID], ...],
                 markers: Tuple[Optional[ChannelID], ...],
                 amplitudes: Tuple[float, ...],
                 offsets: Tuple[float, ...],
                 voltage_transformations: Tuple[Optional[Callable], ...],
                 sample_rate: TimeType,
                 waveforms: Sequence[Waveform] = None):
        """

        Args:
            loop:
            channels:
            markers:
            amplitudes:
            offsets:
            voltage_transformations:
            sample_rate:
            waveforms: These waveforms are sampled and stored in _waveforms. If None the waveforms are extracted from
            loop
        """
        assert len(channels) == len(amplitudes) == len(offsets) == len(voltage_transformations)

        self._channels = tuple(channels)
        self._markers = tuple(markers)
        self._amplitudes = tuple(amplitudes)
        self._offsets = tuple(offsets)
        self._voltage_transformations = tuple(voltage_transformations)

        self._sample_rate = sample_rate

        self._loop = loop

        if waveforms is None:
            waveforms = OrderedDict((node.waveform, None)
                                    for node in loop.get_depth_first_iterator() if node.is_leaf()).keys()
        if waveforms:
            self._waveforms = OrderedDict(zip(waveforms, self._sample_waveforms(waveforms)))
        else:
            self._waveforms = OrderedDict()

    def _sample_empty_channel(self, time: numpy.ndarray) -> Optional[numpy.ndarray]:
        """Override this in derived class to change how """
        return None

    def _sample_empty_marker(self, time: numpy.ndarray) -> Optional[numpy.ndarray]:
        return None

    def _sample_waveforms(self, waveforms: Sequence[Waveform]) -> List[Tuple[Tuple[numpy.ndarray, ...],
                                                                             Tuple[numpy.ndarray, ...]]]:
        sampled_waveforms = []

        time_array, segment_lengths = get_sample_times(waveforms, self._sample_rate)
        for waveform, segment_length in zip(waveforms, segment_lengths):
            wf_time = time_array[:segment_length]

            sampled_channels = []
            for channel, trafo, amplitude, offset in zip(self._channels, self._voltage_transformations,
                                                         self._amplitudes, self._offsets):
                if channel is None:
                    sampled_channels.append(self._sample_empty_channel())
                else:
                    sampled = waveform.get_sampled(channel, wf_time)
                    if trafo is not None:
                        sampled = trafo(sampled)
                    sampled = sampled - offset
                    sampled /= amplitude
                    sampled_channels.append(waveform.get_sampled(channel, wf_time))

            sampled_markers = []
            for marker in self._markers:
                if marker is None:
                    sampled_markers.append(None)
                else:
                    sampled_markers.append(waveform.get_sampled(marker, wf_time) != 0)

            sampled_waveforms.append((tuple(sampled_channels), tuple(sampled_markers)))
        return sampled_waveforms


class OutOfWaveformMemoryException(Exception):

    def __str__(self) -> str:
        return "Out of memory error adding waveform to waveform memory."


class ChannelNotFoundException(Exception):
    def __init__(self, channel):
        self.channel = channel

    def __str__(self) -> str:
        return 'Marker or channel not found: {}'.format(self.channel)
