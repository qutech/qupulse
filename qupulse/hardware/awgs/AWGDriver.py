from abc import ABC, abstractmethod


class BaseFeature(ABC):
    def __init__(self):
        self._functionList = []
        directory = dir(self)
        tmp_list = []
        i = 0
        for attr in directory:
            if callable(getattr(self, attr)) and attr[0] != "_":
                tmp_list.append(attr)

    @property
    def function_list(self):
        return tuple(self._functionList)


class FeatureList(ABC):
    def __init__(self):
        self._featureList = []

    def add_feature(self, feature: BaseFeature):
        for function in feature.function_list:
            setattr(self, function, getattr(feature, function))

        self._featureList.append(feature)


class AWGDeviceFeature(ABC, BaseFeature):
    pass


class AWGChannelFeature(ABC, BaseFeature):
    pass


class AWGChannelTupleFeature(ABC, BaseFeature):
    pass


class Program(ABC):
    def __init__(self, name: str, program: Loop):
        self._name = name
        self._program = program
        self._channel_ids = []
        self._marker_ids = []

    @property
    def name(self):
        return self._name

    @property
    def program(self):
        return self._program

    @property
    def channel_ids(self):
        return self._channel_ids

    @property
    def marker_ids(self):
        return self.marker_ids


class AWGProgramManager(ABC):
    @abstractmethod
    def add(self, program: Program):
        pass

    @abstractmethod
    def get(self, name: str) -> Program:
        pass

    @abstractmethod
    def remove(self, name: str):
        pass

    @abstractmethod
    def clear(self):
        pass


class AWGDevice(ABC, FeatureList):
    def __init__(self):
        self._channels = []
        self._channel_groups = []

    def add_feature(self, feature: AWGDeviceFeature):
        super().add_feature(feature)

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
        pass

    @property
    def channels(self):
        return self._channels

    @property
    def channel_group(self):
        return self._channel_groups


class AWGChannelTuple(ABC, FeatureList):
    def __init__(self, channel_tuple_id: int, device: AWGDevice, channels, sample_rate: float,
                 programs: AWGProgramManager):
        self._channel_tuptle_id = channel_tuple_id
        self._device = device
        self._channels = channels
        self._sample_rate = sample_rate
        self._programs = programs

    def add_feature(self, feature: AWGChannelTupleFeature):
        super().add_feature(feature)

    @abstractmethod
    def _send_cmd(self, cmd: str):
        pass

    @abstractmethod
    def _send_querry(self, cmd: str) -> str:
        pass


class AWGChannel(ABC, FeatureList):
    def __init__(self, channel_id: int, device: AWGDevice, channel_tupel: AWGChannelTuple):
        self._channel_id = channel_id
        self._device = device
        self._channel_tupel = channel_tupel

    def add_feature(self, feature: AWGChannelFeature):
        super().add_feature(feature)

    @abstractmethod
    def _send_cmd(self, cmd: str):
        pass

    @abstractmethod
    def _send_querry(self, cmd: str) -> str:
        pass

    @property
    def channel_id(self):
        return self._channel_id

    @property
    def device(self):
        return self._device

    @property
    def channel_tupel(self):
        return self._channel_tupel
