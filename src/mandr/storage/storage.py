from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Generator

    from mandr.item import Item
    from mandr.storage import URI


class Storage(ABC):
    @abstractmethod
    def __contains__(self, key: URI) -> bool:
        """ """

    @abstractmethod
    def getitem(self, key: URI) -> Item:
        """
        Raises
        ------
        KeyError
        """

    @abstractmethod
    def setitem(self, key: URI, item: Item):
        """ """

    @abstractmethod
    def delitem(self, key: URI):
        """
        Raises
        ------
        KeyError
        """

    @abstractmethod
    def keys(self) -> Generator[URI, None, None]:
        """ """

    @abstractmethod
    def items(self) -> Generator[tuple[URI, Item], None, None]:
        """ """
