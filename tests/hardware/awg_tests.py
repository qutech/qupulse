# Warnungen ignorieren
import warnings

warnings.simplefilter("ignore", UserWarning)

from qupulse.hardware.awgs.AWGDriver import *
import unittest
from typing import TypeVar, Generic, List

print("--Device Test:")


class TestDevice(AWGDevice):
    def __init__(self):
        super().__init__()
        self.initialize()
        self._send_cmd("TestCMD")

    def initialize(self):
        print("initialize")

    def _send_cmd(self, cmd: str):
        print("send cmd: " + cmd)

    def _send_query(self, cmd: str) -> str:
        print("send query")
        return "test String"

    def cleanup(self):
        pass


class DeviceTestFeature(AWGDeviceFeature):
    def device_test_feature_methode1(self):
        print("feature1methode1 works")

    def device_test_feature_methode2(self):
        print("feature1methode2 works")


deviceTestFeature = DeviceTestFeature()
device = TestDevice()

print("Dir(device): ")
print(dir(device))
device.add_feature(deviceTestFeature)
print("Dir(device) mit Feature: ")
print(dir(device))

device.initialize()
device.device_test_feature_methode1()
device.device_test_feature_methode2()

print("")


class Loop:
    pass


testLoop = Loop()


class TestProgram(Program):
    def __init__(self, name: str, loop: Loop):
        super().__init__(name, loop)


testProgram = TestProgram("TestProgram", testLoop)


class TestProgramManager(AWGProgramManager):
    def add(self, program: Program):
        print("Test " + Program)

    def get(self):
        pass

    def remove(self, name: str):
        print("remove")

    def clear(self):
        print("clear")


testProgramManager = TestProgramManager()

print("--ChannelTupelTest:")


class TestChannelTuple(AWGChannelTuple):
    def __init__(self):
        super().__init__(1, device, 8)

    def _send_cmd(self, cmd: str):
        print("send cmd: " + cmd)

    def _send_query(self, cmd: str) -> str:
        print("send query")
        return str

    def sample_rate(self, sample_rate: float):
        pass


channelTupel = TestChannelTuple()


class ChannelTupelTestFeature(AWGChannelTupleFeature):
    def channel_tupel_test_feature_methode1(self):
        print("ChannelTupelTestFeatureMethode1 works")

    def channel_tupel_test_feature_methode2(self):
        print("ChannelTupelTestFeatureMethode2 works")


channelTupelTestFeature = ChannelTupelTestFeature()

channelTupel.add_feature(channelTupelTestFeature)
print("dir(channel):")
print(dir(channelTupel))
channelTupel.add_feature(channelTupelTestFeature)
print("dir(channel):")
print(dir(channelTupel))

channelTupel.channel_tupel_test_feature_methode1()
channelTupel.channel_tupel_test_feature_methode2()

print("")


class TestChannel(AWGChannel):
    def __init__(self, channel_id: int):
        super().__init__(channel_id, device)

    def _send_cmd(self, cmd: str):
        print("send cmd: " + cmd)

    def _send_query(self, cmd: str) -> str:
        print("send query: " + cmd)
        return cmd


print("--ChannelTest:")


class ChannelTestFeature(AWGChannelFeature):
    def channel_test_feature_methode1(self):
        print("ChannelTestFeatureMethode1 works")

    def channel_test_feature_methode2(self):
        print("ChannelTestFeatureMethode2 works")


class SynchronizeChannels(AWGChannelFeature):
    def synchronize(self, test: List[AWGChannel]):
        print("ChannelSynchronisieren")


channel = TestChannel(1)
channelTestFeature = ChannelTestFeature()

print("dir(channel):")
print(dir(channel))
channel.add_feature(channelTestFeature)
print("dir(channel):")
print(dir(channel))
channel.channel_test_feature_methode1()
channel.channel_test_feature_methode2()


class TestAWGDriver(unittest.TestCase):
    def TestDeviceAddFeature(self):
        pass


test_list = List[AWGChannel]

testChannelList = [TestChannel(0), TestChannel(1), TestChannel(2), TestChannel(3), TestChannel(4), TestChannel(5),
                   TestChannel(6), TestChannel(7)]
