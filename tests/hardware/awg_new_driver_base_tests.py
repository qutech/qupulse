from typing import Callable, Iterable, Optional, Set, Tuple
import unittest
import warnings

from qupulse import ChannelID
from qupulse._program._loop import Loop
from qupulse.hardware.feature_awg.base import AWGDevice, AWGChannel, AWGChannelTuple, AWGMarkerChannel
from qupulse.hardware.feature_awg.features import ChannelSynchronization, ProgramManagement, VoltageRange, \
    AmplitudeOffsetHandling
from qupulse.utils.types import Collection

warnings.simplefilter("ignore", UserWarning)



########################################################################################################################
# Example Features
########################################################################################################################

class TestSynchronizeChannelsFeature(ChannelSynchronization):
    def __init__(self, device: "TestAWGDevice"):
        super().__init__()
        self._parent = device

    def synchronize_channels(self, group_size: int) -> None:
        """Forwarding call to TestAWGDevice"""
        self._parent.synchronize_channels(group_size)


class TestVoltageRangeFeature(VoltageRange):
    def __init__(self, channel: "TestAWGChannel"):
        super().__init__()
        self._parent = channel

    @property
    def offset(self) -> float:
        """Get offset of TestAWGChannel"""
        return self._parent._offset

    @offset.setter
    def offset(self, offset: float) -> None:
        """Set offset of TestAWGChannel"""
        self._parent._offset = offset

    @property
    def amplitude(self) -> float:
        """Get amplitude of TestAWGChannel"""
        return self._parent._amplitude

    @amplitude.setter
    def amplitude(self, amplitude: float) -> None:
        """Set amplitude of TestAWGChannel"""
        self._parent._amplitude = amplitude

    @property
    def amplitude_offset_handling(self) -> str:
        """Get amplitude-offset-handling of TestAWGChannel"""
        return self._parent._ampl_offs_handling

    @amplitude_offset_handling.setter
    def amplitude_offset_handling(self, ampl_offs_handling: str) -> None:
        """Set amplitude-offset-handling of TestAWGChannel"""
        self._parent._ampl_offs_handling = ampl_offs_handling


class TestProgramManagementFeature(ProgramManagement):
    def __init__(self):
        super().__init__()
        self._programs = {}
        self._armed_program = None

    def upload(self, name: str, program: Loop, channels: Tuple[Optional[ChannelID], ...],
               marker_channels: Tuple[Optional[ChannelID], ...], voltage_transformation: Tuple[Optional[Callable], ...],
               force: bool = False) -> None:
        if name in self._programs:
            raise KeyError("Program with name \"{}\" is already on the instrument.".format(name))
        self._programs[name] = program

    def remove(self, name: str) -> None:
        if self._armed_program == name:
            raise RuntimeError("Cannot remove program, when it is armed.")
        if name not in self._programs:
            raise KeyError("Unknown program: {}".format(name))
        del self._programs[name]

    def clear(self) -> None:
        if self._armed_program is not None:
            raise RuntimeError("Cannot clear programs, with an armed program.")
        self._programs.clear()

    def arm(self, name: Optional[str]) -> None:
        self._armed_program = name

    @property
    def programs(self) -> Set[str]:
        return set(self._programs.keys())


########################################################################################################################
# Device & Channels
########################################################################################################################

class TestAWGDevice(AWGDevice):
    def __init__(self, name: str):
        super().__init__(name)

        # Add feature to this object (self)
        # During this call, the function of the feature is dynamically added to this object
        self.add_feature(TestSynchronizeChannelsFeature(self))

        self._channels = [TestAWGChannel(i, self) for i in range(8)]  # 8 channels
        self._channel_tuples = []

        # Call the feature function, with the feature's signature
        # Default channel synchronization with a group size of 2
        self[ChannelSynchronization].synchronize_channels(2)

    def cleanup(self) -> None:
        """This will be called automatically in __del__"""
        self._channels.clear()
        self._channel_tuples.clear()

    @property
    def channels(self) -> Collection["TestAWGChannel"]:
        return self._channels

    @property
    def marker_channels(self) -> Collection[AWGMarkerChannel]:
        return []

    @property
    def channel_tuples(self) -> Collection["TestAWGChannelTuple"]:
        return self._channel_tuples

    def synchronize_channels(self, group_size: int) -> None:
        """Implementation of the feature's , but you can also call it directly"""
        if group_size not in [2, 4, 8]:  # Allowed group sizes
            raise ValueError("Invalid group size for channel synchronization")

        self._channel_tuples.clear()
        tmp_channel_tuples = [[] for i in range(len(self._channels) // group_size)]

        # Preparing the channel structure
        for i, channel in enumerate(self._channels):
            tmp_channel_tuples[i // group_size].append(channel)

        # Create channel tuples with its belonging channels and refer to their parent tuple
        for i, tmp_channel_tuple in enumerate(tmp_channel_tuples):
            channel_tuple = TestAWGChannelTuple(i, self, tmp_channel_tuple)
            self._channel_tuples.append(channel_tuple)
            for channel in tmp_channel_tuple:
                channel._set_channel_tuple(channel_tuple)


class TestAWGChannelTuple(AWGChannelTuple):
    def __init__(self, idn: int, device: TestAWGDevice, channels: Iterable["TestAWGChannel"]):
        super().__init__(idn)

        # Add feature to this object (self)
        # During this call, the function of the feature is dynamically added to this object
        self.add_feature(TestProgramManagementFeature())

        self._device = device
        self._channels = tuple(channels)
        self._sample_rate = 12.456  # default value

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, sample_rate: float) -> None:
        self._sample_rate = sample_rate

    @property
    def device(self) -> TestAWGDevice:
        return self._device

    @property
    def channels(self) -> Collection["TestAWGChannel"]:
        return self._channels

    @property
    def marker_channels(self) -> Collection[AWGMarkerChannel]:
        return []


class TestAWGChannel(AWGChannel):
    def __init__(self, idn: int, device: TestAWGDevice):
        super().__init__(idn)

        # Add feature to this object (self)
        # During this call, all functions of the feature are dynamically added to this object
        self.add_feature(TestVoltageRangeFeature(self))

        self._device = device
        self._channel_tuple = None
        self._offset = 0.0
        self._amplitude = 5.0
        self._ampl_offs_handling = AmplitudeOffsetHandling.IGNORE_OFFSET

    @property
    def device(self) -> TestAWGDevice:
        return self._device

    @property
    def channel_tuple(self) -> Optional[TestAWGChannelTuple]:
        return self._channel_tuple

    def _set_channel_tuple(self, channel_tuple: TestAWGChannelTuple) -> None:
        self._channel_tuple = channel_tuple


class TestBaseClasses(unittest.TestCase):
    def setUp(self):
        self.device_name = "My device"
        self.device = TestAWGDevice(self.device_name)

    def test_device(self):
        self.assertEqual(self.device.name, self.device_name, "Invalid name for device")
        self.assertEqual(len(self.device.channels), 8, "Invalid number of channels")
        self.assertEqual(len(self.device.marker_channels), 0, "Invalid number of marker channels")
        self.assertEqual(len(self.device.channel_tuples), 4, "Invalid default channel tuples for device")

    def test_channels(self):
        for i, channel in enumerate(self.device.channels):
            self.assertEqual(channel.idn, i), "Invalid channel id"
            self.assertEqual(channel[VoltageRange].offset, 0, "Invalid default offset for channel {}".format(i))
            self.assertEqual(channel[VoltageRange].amplitude, 5.0,
                             "Invalid default amplitude for channel {}".format(i))

            offs = -0.1 * i
            ampl = 0.5 + 3 * i
            channel[VoltageRange].offset = offs
            channel[VoltageRange].amplitude = ampl

            self.assertEqual(channel[VoltageRange].offset, offs, "Invalid offset for channel {}".format(i))
            self.assertEqual(channel[VoltageRange].amplitude, ampl, "Invalid amplitude for channel {}".format(i))

    def test_channel_tuples(self):
        for group_size in [2, 4, 8]:
            self.device[ChannelSynchronization].synchronize_channels(group_size)

            self.assertEqual(len(self.device.channel_tuples), 8 // group_size, "Invalid number of channel tuples")

            # Check if channels and channel tuples are connected right
            for i, channel in enumerate(self.device.channels):
                self.assertEqual(channel.channel_tuple.idn, i // group_size,
                                 "Invalid channel tuple {} for channel {}".format(channel.channel_tuple.idn, i))
                self.assertTrue(channel in channel.channel_tuple.channels,
                                "Channel {} not in its parent channel tuple {}".format(i, channel.channel_tuple.idn))

        self.assertEqual(len(self.device.channel_tuples), 1, "Invalid number of channel tuples")

    def test_error_handling(self):
        with self.assertRaises(ValueError):
            self.device[ChannelSynchronization].synchronize_channels(3)

        with self.assertRaises(KeyError):
            self.device.add_feature(TestSynchronizeChannelsFeature(self.device))

        with self.assertRaises(TypeError):
            self.device.add_feature(TestProgramManagementFeature())

        with self.assertRaises(TypeError):
            self.device.features[ChannelSynchronization] = None


if __name__ == '__main__':
    unittest.main()
