import warnings
from abc import ABC, abstractmethod
from copy import copy
from typing import TypeVar, Generic, List, Iterable, Dict, Callable


class BaseFeature(ABC):
    """
    Base class for features of `FeatureAble`s.

    Features are classes containing functions which are bound dynamically to the target instance of type `FeatureAble`.
    This ensures that all targets for the same feature are using the same signature for the feature's functions. All
    public callables of a specific feature will be added to the function dictionary. Those functions (in the `functions`
    dictionary) will be automatically added to the specific `FeatureAble` that calls `FeatureAble.add_feature`.
    """

    def __init__(self):
        super().__init__()

        self._functions = self._read_functions()

    def _read_functions(self) -> Dict[str, Callable]:
        """
        Reads the functions of a feature and returns them as a dictionary

        Return:
            Returns dictionary with all functions of the feature
        """
        directory = dir(self)
        function_list = {}
        for attr in directory:
            if callable(getattr(type(self), attr)) and attr[0] != "_":
                if not (attr in function_list):
                    function_list[attr] = getattr(self, attr)
        return function_list

    @property
    def functions(self) -> Dict[str, Callable]:
        """Returns a copy of the dictionary with all public functions of the feature"""
        return copy(self._functions)


FeatureType = TypeVar(BaseFeature)


class FeatureAble(Generic[FeatureType], ABC):
    """Base class for all classes that are able to add features"""

    def __init__(self):
        super().__init__()

        self._features = {}

    @property
    def features(self) -> Dict[str, Callable]:
        """Returns the dictionary with all features of a FeatureAble"""
        return copy(self._features)

    def add_feature(self, feature: FeatureType) -> None:
        """
        The method adds all functions of feature to a dictionary with all functions

        Args:
             feature: A certain feature which functions should be added to the dictionary _features
        """
        if not isinstance(feature, FeatureType):
            raise TypeError("Invalid type for feature")

        for function in feature.function_list:
            if not hasattr(self, function):
                setattr(self, function, getattr(feature, function))
            else:
                warnings.warning(
                    f"Ommiting function \"{function}\": Another attribute with this name already exists.")

        self._features[type(feature).__name__] = feature


class AWGDeviceFeature(BaseFeature, ABC):
    """Base class for features that are used for `AWGDevice`s"""
    pass


class AWGChannelFeature(BaseFeature, ABC):
    """Base class for features that are used for `AWGChannel`s"""
    pass


class AWGChannelTupleFeature(BaseFeature, ABC):
    """Base class for features that are used for `AWGChannelTuple`s"""
    pass


class BaseAWG(FeatureAble[AWGDeviceFeature], ABC):
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

    @abstractmethod
    def cleanup(self) -> None:
        """Function for cleaning up the dependencies of the device"""
        pass

    @property
    def name(self) -> str:
        """Returns the name of a Device as a String"""
        return self._name

    @property
    @abstractmethod
    def channels(self) -> Iterable["BaseAWGChannel"]:
        """Returns a list of all channels of a Device"""
        return self._channels

    @property
    @abstractmethod
    def channel_tuples(self) -> Iterable["BaseAWGChannelTuple"]:
        """Returns a list of all channel tuples of a list"""
        pass


class BaseAWGChannelTuple(FeatureAble[AWGChannelTupleFeature], ABC):
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
    def sample_rate(self) -> float:
        """Returns the sample rate of a channel tuple as a float"""
        pass

    @property
    def idn(self) -> int:
        """Returns the identification number of a channel tuple"""
        return self._idn

    @property
    @abstractmethod
    def device(self) -> BaseAWG:
        """Returns the device which the channel tuple belong to"""
        pass

    @property
    @abstractmethod
    def channels(self) -> Iterable["BaseAWGChannel"]:
        """Returns a list of all channels of the channel tuple"""
        pass


class BaseAWGChannel(FeatureAble[AWGChannelFeature], ABC):
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
        pass

    @property
    @abstractmethod
    def channel_tuple(self) -> BaseAWGChannelTuple:
        """Returns the channel tuple which a channel belongs to"""
        pass

    @abstractmethod
    def _set_channel_tuple(self, channel_tuple: BaseAWGChannelTuple):
        """
        Sets the channel tuple which a channel belongs to
        Args:
            channel_tuple: reference to the channel tuple
        """
        pass
