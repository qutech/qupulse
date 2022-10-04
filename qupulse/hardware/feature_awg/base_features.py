from types import MappingProxyType
from typing import Callable, Generic, Mapping, Optional, Type, TypeVar
from abc import ABC

__all__ = ["Feature", "FeatureAble"]


class Feature:
    """
    Base class for features of :class:`.FeatureAble`.
    """
    def __init__(self, target_type: Type["FeatureAble"]):
        self._target_type = target_type

    @property
    def target_type(self) -> Type["FeatureAble"]:
        return self._target_type


FeatureType = TypeVar("FeatureType", bound=Feature)
GetItemFeatureType = TypeVar("GetItemFeatureType", bound=Feature)


class FeatureAble(Generic[FeatureType]):
    """
    Base class for all types that are able to handle features. The features are saved in a dictionary and the methods
    can be called with the `__getitem__`-operator.
    """

    def __init__(self):
        super().__init__()
        self._features = {}

    def __getitem__(self, feature_type: Type[GetItemFeatureType]) -> GetItemFeatureType:
        if isinstance(feature_type, str):
            return self._features[feature_type]
        if not isinstance(feature_type, type):
            raise TypeError("Expected type-object as key, got \"{ftt}\" instead".format(
                ftt=type(feature_type).__name__))
        key_type = _get_base_feature_type(feature_type)
        if key_type is None:
            raise TypeError("Unexpected type of feature: {ft}".format(ft=feature_type.__name__))
        if key_type not in self._features:
            raise KeyError("Could not get feature for type: {ft}".format(ft=feature_type.__name__))
        return self._features[key_type]

    def add_feature(self, feature: FeatureType) -> None:
        """
        The method adds the feature to a Dictionary with all features

        Args:
             feature: A certain feature which functions should be added to the dictionary _features
        """
        feature_type = _get_base_feature_type(type(feature))
        if feature_type is None:
            raise TypeError("Unexpected type of feature: {ft}".format(ft=type(feature).__name__))
        if not isinstance(self, feature.target_type):
            raise TypeError("Features with type \"{ft}\" belong to \"{tt}\"-objects".format(
                ft=type(feature).__name__, tt=feature.target_type.__name__))
        if feature_type in self._features:
            raise KeyError(self, "Feature with type \"{ft}\" already exists".format(ft=feature_type.__name__))
        self._features[feature_type] = feature
        # Also adding the feature with the string as the key. With this you can you the name as a string for __getitem__
        self._features[feature_type.__name__] = feature

    @property
    def features(self) -> Mapping[FeatureType, Callable]:
        """Returns the dictionary with all features of a FeatureAble"""
        return MappingProxyType(self._features)


def _get_base_feature_type(feature_type: Type[Feature]) -> Type[Optional[Feature]]:
    """
    This function searches for the second inheritance level under `Feature` (i.e. level under `AWGDeviceFeature`,
    `AWGChannelFeature` or `AWGChannelTupleFeature`). This is done to ensure, that nobody adds the same feature
    twice, but with a type of a different inheritance level as key.

    Args:
        feature_type: Type of the feature

    Returns:
        Base type of the feature_type, two inheritance levels under `Feature`
    """
    if not issubclass(feature_type, Feature):
        return type(None)

    # Search for base class on the inheritance line of Feature
    for base in feature_type.__bases__:
        if issubclass(base, Feature):
            result_type = base
            break
    else:
        return type(None)

    if Feature in result_type.__bases__:
        return feature_type
    else:
        return _get_base_feature_type(result_type)
