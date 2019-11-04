from abc import ABC, abstractmethod


class AWGDevice(ABC):
    channels = []
    channel_groups = []

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def synchronoize_channels(self, group: int):
        pass

    @abstractmethod
    def _send_cmd(self, cmd: str):
        pass

    @abstractmethod
    def _send_querry(self, cmd: str) -> str:
        # Wie wird der Return vom String umgesetzt
        pass

class AWGChannelTuple(ABC):
    def __init__(self, channel_tuple_id, device: AWGDevice, channels, sample_rate, programs):
        self._channel_tuptle_id = channel_tuple_id
        self._device = device
        self._channels = channels
        self._sample_rate = sample_rate
        self._programs = programs

    @abstractmethod
    def _send_cmd(self, cmd: str):
        pass

    @abstractmethod
    def _send_querry(self, cmd: str) -> str:
        # Wie wird der Return vom String umgesetzt
        pass


class AWGChannel(ABC):
    def __init__(self, channel_id: int, device: AWGDevice, channel_tupel: AWGChannelTuple):
        self._channel_id = channel_id
        self._device = device
        self._channel_tupel = channel_tupel

    @abstractmethod
    def _send_cmd(self, cmd: str):
        pass

    @abstractmethod
    def _send_querry(self, cmd: str) -> str:
        # Was passiert wenn was anderes als ein String returnt wird?
        pass

    # muss die Methode dann auch abstrakt sein?
    @property
    def channel_id(self):
        return self._channel_id

    # braucht channelId einen Setter?

class AWGProgrammManager(ABC):
    @abstractmethod
    def add(self, program: Programm):
        pass

    @abstractmethod
    def get(self, name: str) -> Programm:
        # Wie macht man die rückgabe von Programm?
        pass

    @abstractmethod
    def remove(self, name: str):
        pass

    @abstractmethod
    def clear(self):
        pass


# Basiskalssen

class BaseFeature(ABC):
    pass


class AWGDeviceFeature(AWGDevice, ABC, BaseFeature):
    pass


class AWGChannelFeature(AWGChannel, ABC, BaseFeature):
    pass


class AWGChannelTupleFeature(AWGChannelTuple, ABC, BaseFeature):
    pass
