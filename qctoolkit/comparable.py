from abc import ABCMeta, abstractproperty
from typing import Any


__all__ = ["Comparable"]


class Comparable(metaclass=ABCMeta):
    """An object that can be queried for equality with other Comparable object."""

    @abstractproperty
    def _compare_key(self) -> Any:
        """Returns a unique key used in comparison and hashing operations.

        The key must describe the essential properties of the object. Two objects are equal iff their keys are identical.
        """

    def __hash__(self) -> int:
        """Returns a hash value of the comparable object."""
        return hash(self._compare_key)

    def __eq__(self, other: Any) -> bool:
        """True, if other is equal to self."""
        return isinstance(other, self.__class__) and self._compare_key == other._compare_key

    def __ne__(self, other: Any) -> bool:
        """True, if other is not equal to self."""
        return not self == other