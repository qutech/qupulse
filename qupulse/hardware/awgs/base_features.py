import typing
from typing import TypeVar, Generic, Dict, Callable
import warnings
from abc import ABC
from copy import copy


class Feature(ABC):
    """
    Base class for features of `FeatureAble`s.
    """
    pass


FeatureType = TypeVar("FeatureType", bound=Feature)


class FeatureAble(Generic[FeatureType], ABC):
    """
    Base class for all classes that are able to add features. The features are saved in a dictonary and the methods
    can be called with __getitem__.
    """

    def __init__(self):
        super().__init__()

        self._features = {}

    def __getitem__(self, feature_type: typing.Type[FeatureType]) -> FeatureType:
        return self._features[feature_type]

    def add_feature(self, feature: FeatureType) -> None:
        """
        The method adds the feature to a Dictionary with all features

        Args:
             feature: A certain feature which functions should be added to the dictionary _features
        """
        if not isinstance(feature, Feature):
            raise TypeError("Invalid type for feature")

        self._features[type(feature)] = feature

    @property
    def features(self) -> Dict[FeatureType, Callable]:
        """Returns the dictionary with all features of a FeatureAble"""
        return copy(self._features)
