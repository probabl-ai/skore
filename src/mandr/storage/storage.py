from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Generator

    from mandr.item import Item
    from mandr.storage import URI


class Storage(ABC):
    @abstractmethod
    def __iter__(self) -> Generator[URI, None, None]:
        """ """

    @abstractmethod
    def contains(self, uri: URI, key: str) -> bool:
        """ """

    @abstractmethod
    def getitem(self, uri: URI, key: str) -> Item:
        """
        Raises
        ------
        KeyError
        """

    @abstractmethod
    def setitem(self, uri: URI, key: str, item: Item):
        """ """

    @abstractmethod
    def delitem(self, uri: URI, key: str):
        """
        Raises
        ------
        KeyError
        """

    @abstractmethod
    def keys(self, uri: URI) -> Generator[str, None, None]:
        """ """

    @abstractmethod
    def items(self, uri: URI) -> Generator[tuple[str, Item], None, None]:
        """ """
