import typing
import abc
import inspect
import numbers
import fractions
import collections
import itertools
from collections.abc import Mapping as ABCMapping

import numpy

__all__ = ["MeasurementWindow", "ChannelID", "HashableNumpyArray", "TimeType", "time_from_float", "DocStringABCMeta",
           "OrderedSet", "ConstantDictView"]

MeasurementWindow = typing.Tuple[str, numbers.Real, numbers.Real]
ChannelID = typing.Union[str, int]
TimeType = fractions.Fraction


def time_from_float(time: float, absolute_error: float=1e-12) -> TimeType:
    return fractions.Fraction(time).limit_denominator(int(1/absolute_error))


class DocStringABCMeta(abc.ABCMeta):
    def __new__(mcls, classname, bases, cls_dict):
        cls = super().__new__(mcls, classname, bases, cls_dict)

        abstract_bases = tuple(base
                               for base in reversed(inspect.getmro(cls))
                               if hasattr(base, '__abstractmethods__'))[:-1]

        for name, member in cls_dict.items():
            if not getattr(member, '__doc__'):
                if isinstance(member, property):
                    member_type = ':py:attr:'
                else:
                    member_type = ':func:'

                for base in abstract_bases:
                    if name in base.__dict__ and name in base.__abstractmethods__:
                        base_member = getattr(base, name)

                        if member is base_member or not base_member.__doc__:
                            continue

                        base_member_name = '.'.join([base.__module__, base.__qualname__, name])
                        member.__doc__ = 'Implements {}`~{}`.'.format(member_type, base_member_name)
                        break
        return cls


class HashableNumpyArray(numpy.ndarray):
    """Make numpy arrays hashable.

    Example usage:
    my_array = np.zeros([1, 2, 3, 4])
    hashable = my_array.view(HashableNumpyArray)
    """
    def __hash__(self):
        return hash(self.tobytes())


class ConstantDictView(ABCMapping):
    """Immutable view on a dict"""
    __slots__ = ("__getitem__", "__contains__", "keys", "values", "items", "__iter__", "__len__", "get")

    def __init__(self, dict_obj):
        for slot in self.__slots__:
            setattr(self, slot, getattr(dict_obj, slot))


class OrderedSet(collections.OrderedDict, collections.MutableSet):
    def update(self, *args, **kwargs):
        if kwargs:
            raise TypeError("update() takes no keyword arguments")

        update = super(collections.OrderedDict, self).update
        for s in args:
            update(itertools.zip_longest(s, ()))

    def add(self, elem):
        self[elem] = None

    def discard(self, elem):
        self.pop(elem, None)

    def remove(self, element):
        self.pop(element)

    def __le__(self, other):
        return self.keys() <= other.keys()

    def __lt__(self, other):
        return self.keys() < other.keys()

    def __ge__(self, other):
        return self.keys() >= other.keys()

    def __gt__(self, other):
        return self.keys() > other.keys()

    def __eq__(self, other):
        return self.keys() == other.keys()

    def __ne__(self, other):
        return self.keys() != other.keys()

    def __repr__(self):
        return 'OrderedSet([%s])' % (', '.join(map(repr, self.keys())))

    def __str__(self):
        return '{%s}' % (', '.join(map(repr, self.keys())))
