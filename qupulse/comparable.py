"""This module defines the abstract Comparable class."""
from abc import abstractmethod
from typing import Hashable, Any

from qupulse.utils.types import DocStringABCMeta


__all__ = ["Comparable"]


class Comparable(metaclass=DocStringABCMeta):
    """An object that can be queried for equality with other Comparable objects.

    Subclasses must override the abstract property _compare_key which shall provide some object
    natively equatable in Python (e.g., strings, numbers, tuples containing those, etc..).
    Comparable provides implementations of the hashing function as well as the equals and not-equals
    operators based on comparison of this key.
    """
    __slots__ = ()

    @property
    @abstractmethod
    def compare_key(self) -> Hashable:
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
