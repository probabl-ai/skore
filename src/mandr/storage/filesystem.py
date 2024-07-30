"""Persistent storage over disk based on the diskcache library."""

from __future__ import annotations

from typing import TYPE_CHECKING

from diskcache import Cache

from mandr.storage.storage import Storage

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Generator

    from mandr.item import Item
    from mandr.storage.storage import URI


class FileSystem(Storage):
    """Persistent storage implementation over disk based on the diskcache library."""

    def __init__(self, *, directory: Path | None = None):
        self.cache = Cache(directory)

    def __contains__(self, key: URI) -> bool:
        """Return True if the storage has the specified key, else False."""
        return key in self.cache

    def __iter__(self) -> Generator[URI, None, None]:
        """Yield the keys."""
        yield from self.cache.iterkeys()

    def getitem(self, key: URI) -> Item:
        """Return the item for te specified key.

        Raises
        ------
        KeyError
            If the storage doesn't have the specified key.
        """
        return self.cache[key]

    def setitem(self, key: URI, item: Item):
        """Set the item for the specified key."""
        self.cache[key] = item

    def delitem(self, key: URI):
        """Delete the specified key and its item.

        Raises
        ------
        KeyError
            If the storage doesn't have the specified key.
        """
        del self.cache[key]

    def keys(self) -> Generator[URI, None, None]:
        """Yield the keys."""
        yield from self

    def items(self) -> Generator[tuple[URI, Item], None, None]:
        """Yield the pairs (key, item)."""
        for key in self.cache.iterkeys():
            yield (key, self.cache[key])
