"""Persistent storage over disk based on the diskcache library."""

from __future__ import annotations

from typing import TYPE_CHECKING

from diskcache import Cache

from skore.storage.storage import Storage

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Iterator

    from skore.item import Item


class DirectoryDoesNotExist(Exception):
    """Directory does not exist."""


class FileSystem(Storage):
    """Persistent storage implementation over disk based on the diskcache library."""

    def __init__(self, *, directory: Path):
        if not directory.exists():
            raise DirectoryDoesNotExist(f"Directory {directory} does not exist.")
        self.cache = Cache(directory)

    def __contains__(self, key: str) -> bool:
        """Return True if the storage has the specified key, else False."""
        return key in self.cache

    def __iter__(self) -> Iterator[str]:
        """Yield the keys."""
        yield from self.cache.iterkeys()

    def getitem(self, key: str) -> Item:
        """Return the item for te specified key.

        Raises
        ------
        KeyError
            If the storage doesn't have the specified key.
        """
        return self.cache[key]

    def setitem(self, key: str, item: Item):
        """Set the item for the specified key."""
        self.cache[key] = item

    def delitem(self, key: str):
        """Delete the specified key and its item.

        Raises
        ------
        KeyError
            If the storage doesn't have the specified key.
        """
        del self.cache[key]

    def keys(self) -> Iterator[str]:
        """Yield the keys."""
        yield from self

    def items(self) -> Iterator[tuple[str, Item]]:
        """Yield the pairs (key, item)."""
        for key in self.cache.iterkeys():
            yield (key, self.cache[key])
