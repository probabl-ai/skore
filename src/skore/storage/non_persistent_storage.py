"""Non-persistent storage over RAM based on dict."""

from __future__ import annotations

from typing import TYPE_CHECKING

from skore.storage.storage import Storage

if TYPE_CHECKING:
    from typing import Generator

    from skore.item import Item
    from skore.storage.storage import URI


class NonPersistentStorage(Storage):
    """Non-persistent storage over RAM based on dict class."""

    def __init__(self, *, content: dict = None):
        self.content = content or {}

    def __contains__(self, key: URI) -> bool:
        """Return True if the storage has the specified key, else False."""
        return key in self.content

    def __iter__(self) -> Generator[URI, None, None]:
        """Yield the keys."""
        yield from self.content.keys()

    def getitem(self, key: URI) -> Item:
        """Return the item for te specified key.

        Raises
        ------
        KeyError
            If the storage doesn't have the specified key.
        """
        return self.content[key]

    def setitem(self, key: URI, item: Item):
        """Set the item for the specified key."""
        self.content[key] = item

    def delitem(self, key: URI):
        """Delete the specified key and its item.

        Raises
        ------
        KeyError
            If the storage doesn't have the specified key.
        """
        del self.content[key]

    def keys(self) -> Generator[URI, None, None]:
        """Yield the keys."""
        yield from self

    def items(self) -> Generator[tuple[URI, Item], None, None]:
        """Yield the pairs (key, item)."""
        yield from self.content.items()
