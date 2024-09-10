"""Storage interface used to store key-item pairs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Iterator


class Storage(ABC):
    """Storage interface used to store key-item pairs."""

    @abstractmethod
    def __contains__(self, key: str) -> bool:
        """Return True if the storage has the specified key, else False."""

    @abstractmethod
    def __iter__(self) -> Iterator[str]:
        """Yield the keys."""

    @abstractmethod
    def getitem(self, key: str) -> Any:
        """Return the item for the specified key.

        Raises
        ------
        KeyError
            If the storage doesn't have the specified key.
        """

    @abstractmethod
    def setitem(self, key: str, item: Any):
        """Set the item for the specified key."""

    @abstractmethod
    def delitem(self, key: str):
        """Delete the specified key and its item.

        Raises
        ------
        KeyError
            If the storage doesn't have the specified key.
        """

    @abstractmethod
    def keys(self) -> Iterator[str]:
        """Yield the keys."""

    @abstractmethod
    def items(self) -> Iterator[tuple[str, Any]]:
        """Yield the pairs (key, item)."""
