from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from diskcache import Cache

from mandr.storage.storage import Storage

if TYPE_CHECKING:
    from typing import Generator
    from mandr.item import Item
    from mandr.storage.storage import URI


class FileSystem(Storage):
    """
    Persistent storage over disk.
    """

    def __init__(self, *, directory: Path | None = None):
        self.cache = Cache(directory)

    def __contains__(self, key: URI) -> bool:
        return (key in self.cache)

    def getitem(self, key: URI) -> Item:
        return self.cache[key]

    def setitem(self, key: URI, item: Item):
        self.cache[key] = item

    def delitem(self, key: URI):
        del self.cache[key]

    def keys(self) -> Generator[URI, None, None]:
        yield from self.cache.iterkeys()

    def items(self) -> Generator[tuple[URI, Item], None, None]:
        for key in self.cache.iterkeys():
            yield (key, self.cache[key])
