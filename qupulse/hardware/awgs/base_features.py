import warnings
from abc import ABC
from copy import copy
from typing import TypeVar, Generic, Dict, Callable


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
        function_list: Dict[str, Callable] = {}
        for attr in directory:
            if callable(getattr(type(self), attr)) and attr[0] != "_":
                if not (attr in function_list):
                    function_list[attr] = getattr(self, attr)
        return function_list

    @property
    def functions(self) -> Dict[str, Callable]:
        """Returns a copy of the dictionary with all public functions of the feature"""
        return copy(self._functions)


FeatureType = TypeVar("FeatureType", bound=BaseFeature)


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
        if not isinstance(feature, BaseFeature):
            raise TypeError("Invalid type for feature")

        for function in feature.functions:
            if not hasattr(self, function):
                setattr(self, function, getattr(feature, function))
            else:
                warnings.warn(f"Omitting function \"{function}\": Another attribute with this name already exists.")

        self._features[type(feature).__name__] = feature
