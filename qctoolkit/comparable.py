"""This module defines the abstract Comparable class."""
from abc import ABCMeta, abstractproperty
from typing import Any


__all__ = ["Comparable"]


class Comparable(metaclass=ABCMeta):
    """An object that can be queried for equality with other Comparable objects.

    Subclasses must override the abstract property _compare_key which shall provide some object
    natively equatable in Python (e.g., strings, numbers, tuples containing those, etc..).
    Comparable provides implementations of the hashing function as well as the equals and not-equals
    operators based on comparison of this key.
    """

    @abstractproperty
    def compare_key(self) -> Any:
        """Return a unique key used in comparison and hashing operations.

        The key must describe the essential properties of the object.
        Two objects are equal iff their keys are identical.
        """

    def __hash__(self) -> int:
        """Return a hash value of this Comparable object."""
        return hash(self.compare_key)

    def __eq__(self, other: Any) -> bool:
        """True, if other is equal to this Comparable object."""
        return isinstance(other, self.__class__) and self.compare_key == other.compare_key

    def __ne__(self, other: Any) -> bool:
        """True, if other is not equal to this Comparable object."""
        return not self == other


def extend_comparison(cls):
    if not hasattr(cls, '__lt__'):
        raise ValueError('Class does not implement __lt__')

    def __eq__(self, other):
        return not self < other and not other < self

    def __ne__(self, other):
        return self < other or other < self

    def __gt__(self, other):
        return other < self

    def __ge__(self, other):
        return not self < other

    def __le__(self, other):
        return not other < self
    operations = {'__eq__': __eq__,
                  '__ne__': __ne__,
                  '__gt__': __gt__,
                  '__ge__': __ge__,
                  '__le__': __le__}
    for operation_name, operation_func in operations.items():
        if not hasattr(cls, operation_name):
            setattr(cls, operation_name, operation_func)
    return cls
