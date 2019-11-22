from abc import ABC, abstractmethod


class BaseFeature(ABC):
    def __init__(self):
        self._functionList = []


class AWGDeviceFeature(ABC, BaseFeature):
    pass


class AWGChannelFeature(ABC, BaseFeature):
    pass


class AWGChannelTupleFeature(ABC, BaseFeature):
    pass


class Program:
    def __init__(self, name: str, program: Loop):
        self._name = name
        self._program = program
        self._channel_ids = []
        self._marker_ids = []


class AWGProgramManager:
    def add(self, program: Program):
        pass

    def get(self, name: str) -> Program:
        # Wie macht man die rückgabe von Programm?
        pass

    def remove(self, name: str):
        pass

    def clear(self):
        pass


class AWGDevice(ABC):
    def __init__(self):
        self._channels = []
        self._channel_groups = []

        # Code redundanz
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
        # str is not a callable
        for function in feature._functionList:
            #in Liste einfügen



class AWGChannelTuple(ABC):
    def __init__(self, channel_tuple_id: int, device: AWGDevice, channels, sample_rate: float,
                 programs: AWGProgramManager):
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
