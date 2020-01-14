from abc import ABC, abstractmethod
from qupulse._program._loop import Loop
from typing import TypeVar, Generic, List


# TODO: Abstrakte Getter und Setter

class BaseFeature(ABC):
    def __init__(self):
        super().__init__()

        self._function_list = {}

        directory = dir(self)
        i = 0
        for attr in directory:
            if callable(getattr(self, attr)) and attr[0] != "_":
                if not (attr in self._function_list):
                    self._function_list[attr] = getattr(self, attr)
                    # self._functionList.append(attr)

    # TODO: es heisst function_list aber ist ein Dictionary

    @property
    def function_list(self):
        return tuple(self._function_list)


FeatureType = TypeVar(BaseFeature)


class FeatureList(Generic[FeatureType], ABC):
    def __init__(self):
        super().__init__()

        self._feature_list = {}

    @property
    def feature_list(self):
        return self._feature_list

    def add_feature(self, feature: FeatureType):
        for function in feature.function_list:
            setattr(self, function, getattr(feature, function))

        # self._feature_list.append(feature)
        self._feature_list[type(feature).__name__] = feature

    @abstractmethod
    def _send_cmd(self, cmd: str):
        pass

    @abstractmethod
    def _send_query(self, cmd: str) -> str:
        pass


class AWGDeviceFeature(BaseFeature, ABC):
    pass


class AWGChannelFeature(BaseFeature, ABC):
    pass


class AWGChannelTupleFeature(BaseFeature, ABC):
    pass


class Program(ABC):
    def __init__(self, name: str, program: "Loop"):
        super().__init__()

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


class AWGDevice(FeatureList[AWGDeviceFeature], ABC):
    def __init__(self):
        super().__init__()

        self._channels = List["AWGChannel"]  # TODO: "AWGChannel"
        self._channel_groups = List[AWGChannelTuple]

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def cleanup(self):
        pass

    @property
    def channels(self):
        return self._channels

    # TODO: Ueberpruefung in " "
    #@channels.setter
    #@abstractmethod
    #def channels(self, channels: List["AWGChannel"]):
    #    pass

    @property
    def channel_group(self):
        return self._channel_groups


class AWGChannelTuple(FeatureList[AWGChannelTupleFeature], ABC):
    def __init__(self, channel_tuple_id: int, device: AWGDevice, channels):
        super().__init__()

        self._channel_tuple_id = channel_tuple_id
        self._device = device
        self._channels = channels

    #@property
    #@abstractmethod
    #def sample_rate(self):
    #    pass

    #@sample_rate.setter
    #@abstractmethod
    #def channel_tuple(self, sample_rate: float):
    #   pass

    @property
    def channel_tuple_id(self):
        return self._channel_tuple_id

    @property
    def device(self):
        return self._device

    @property
    def channels(self):
        return self._channels


class AWGChannel(FeatureList[AWGChannelFeature], ABC):
    def __init__(self, channel_id: int, device: AWGDevice):
        super().__init__()
        self._channel_id = channel_id
        self._device = device
        self._channel_tuple = None

    @property
    def channel_id(self):
        return self._channel_id

    @property
    def device(self):
        return self._device

    @property
    def channel_tuple(self):
        return self._channel_tupel

    # TODO: @channel_tuple.setter da hat kann es keinen _ davor haben. -> Namensgebungs?

    @channel_tuple.setter
    def channel_tuple(self, channel_tuple: AWGChannelTuple):
        self._channel_tuple = channel_tuple
