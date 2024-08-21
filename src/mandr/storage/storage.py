"""Storage interface used to store key-item pairs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Generator

    from mandr.item import Item
    from mandr.storage import URI


class Storage(ABC):
    """Storage interface used to store key-item pairs."""

    @abstractmethod
    def __contains__(self, key: URI) -> bool:
        """Return True if the storage has the specified key, else False."""

    @abstractmethod
    def __iter__(self) -> Generator[URI, None, None]:
        """Yield the keys."""

    @abstractmethod
    def getitem(self, key: URI) -> Item:
        """Return the item for the specified key.

        Raises
        ------
        KeyError
            If the storage doesn't have the specified key.
        """

    @abstractmethod
    def setitem(self, key: URI, item: Item):
        """Set the item for the specified key."""

    @abstractmethod
    def delitem(self, key: URI):
        """Delete the specified key and its item.

        Raises
        ------
        KeyError
            If the storage doesn't have the specified key.
        """

    @abstractmethod
    def keys(self) -> Generator[URI, None, None]:
        """Yield the keys."""

    @abstractmethod
    def items(self) -> Generator[tuple[URI, Item], None, None]:
        """Yield the pairs (key, item)."""
