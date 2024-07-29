from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from diskcache import Cache

from mandr.storage.storage import Storage

if TYPE_CHECKING:
    from typing import Generator
    from mandr.item import Item
    from mandr.storage.storage import URI


@dataclass(frozen=True)
class Key:
    prefix: URI
    radical: str


class FileSystem(Storage):
    """
    Persistent storage over disk.
    """

    def __init__(self, *, directory: Path | None = None):
        self.cache = Cache(directory)

    def __iter__(self) -> Generator[URI, None, None]:
        yield from {key.prefix for key in self.cache.iterkeys()}

    def contains(self, uri: URI, key: str) -> bool:
        return Key(uri, key) in self.cache

    def getitem(self, uri: URI, key: str, default: Item = None) -> Item:
        return self.cache[Key(uri, key)]

    def setitem(self, uri: URI, key: str, item: Item):
        self.cache[Key(uri, key)] = item

    def delitem(self, uri: URI, key: str):
        del self.cache[Key(uri, key)]

    def keys(self, uri: URI) -> Generator[str, None, None]:
        for key in self.cache.iterkeys():
            if uri == key.prefix:
                yield key.radical

    def items(self, uri: URI) -> Generator[tuple[str, Item], None, None]:
        for key in self.cache.iterkeys():
            if uri == key.prefix:
                yield (key.radical, self.cache[key])
