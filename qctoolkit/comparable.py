from abc import ABCMeta, abstractproperty
from typing import Any


__all__ = ["Comparable"]


class Comparable(metaclass=ABCMeta):

    @abstractproperty
    def _compare_key(self) -> Any:
        """Return a unique key used in comparison and hashing operations."""

    def __hash__(self) -> int:
        return hash(self._compare_key)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and self._compare_key == other._compare_key

    def __ne__(self, other: Any) -> bool:
        return not self == other