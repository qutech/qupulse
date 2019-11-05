from abc import ABC, abstractmethod


class BaseFeature(ABC):
    pass


class AWGDeviceFeature(ABC, BaseFeature):
    pass


class AWGChannelFeature(ABC, BaseFeature):
    pass


class AWGChannelTupleFeature(ABC, BaseFeature):
    pass


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


class AWGDevice(ABC):
    def __init__(self):
        self._channels = []
        self._channel_groups = []

        self._featureList = []

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

    @abstractmethod
    def add_feature(self, feature: AWGDeviceFeature):
        self._featureList.append(feature)


class AWGChannelTuple(ABC):
    def __init__(self, channel_tuple_id: int, device: AWGDevice, channels, sample_rate: float,
                 programs: AWGProgrammManager):
        self._channel_tuptle_id = channel_tuple_id
        self._device = device
        self._channels = channels
        self._sample_rate = sample_rate
        self._programs = programs

        self._featureList = []

    @abstractmethod
    def _send_cmd(self, cmd: str):
        pass

    @abstractmethod
    def _send_querry(self, cmd: str) -> str:
        # Wie wird der Return vom String umgesetzt
        pass

    @abstractmethod
    def _add_feature(self, feature: AWGChannelFeature):
        self._featureList.append(feature)


class AWGChannel(ABC):
    def __init__(self, channel_id: int, device: AWGDevice, channel_tupel: AWGChannelTuple):
        self._channel_id = channel_id
        self._device = device
        self._channel_tupel = channel_tupel

        self._featureList = []

    @abstractmethod
    def _send_cmd(self, cmd: str):
        pass

    @abstractmethod
    def _send_querry(self, cmd: str) -> str:
        # Was passiert wenn was anderes als ein String returnt wird?
        pass

    @abstractmethod
    def addFeature(self, feature: AWGChannelFeature):
        self._featureList.append(feature)

    # muss die Methode dann auch abstrakt sein?

    # Getter fuer alle Attribute
    @property
    def channel_id(self):
        return self._channel_id

    @property
    def device(self):
        return self.device

    @property
    def channel_tupel(self):
        return self.channel_tupel

    # braucht channelId einen Setter?

# Basisklassen
