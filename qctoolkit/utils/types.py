import typing
import abc
import inspect
import numbers
import fractions

import numpy

__all__ = ["MeasurementWindow", "ChannelID", "HashableNumpyArray", "TimeType", "time_from_float", "ReadOnlyChainMap"]

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


KT, VT = typing.TypeVar('KT'), typing.TypeVar('VT')


class ReadOnlyChainMap(typing.Mapping, typing.Generic[KT, VT]):

    def __init__(self, chain_map: typing.ChainMap) -> None:
        self._chain_map = chain_map

    def __getitem__(self, item: KT) -> VT:
        return self._chain_map[item]

    def __len__(self) -> int:
        return len(self._chain_map)

    def __iter__(self) -> typing.Iterator:
         return iter(self._chain_map)

    def __str__(self) -> str:
        return "ReadOnly{}".format(str(self._chain_map))

    def __repr__(self) -> str:
        return "ReadOnly{}".format(repr(self._chain_map))
