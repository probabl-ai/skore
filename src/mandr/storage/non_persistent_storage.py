from __future__ import annotations

from typing import TYPE_CHECKING

from mandr.storage.storage import Storage

if TYPE_CHECKING:
    from typing import Generator

    from mandr.item import Item
    from mandr.storage.storage import URI


class NonPersistentStorage(Storage):
    """
    Non-persistent storage over dict class.
    """

    def __init__(self, *, content=None):
        self.content = content or {}

    def __iter__(self) -> Generator[URI, None, None]:
        yield from self.content.keys()

    def contains(self, uri: URI, key: str) -> bool:
        return (uri in self.content) and (key in self.content[uri])

    def getitem(self, uri: URI, key: str) -> Item:
        return self.content[uri][key]

    def setitem(self, uri: URI, key: str, item: Item):
        if uri not in self.content:
            self.content[uri] = {key: item}
        else:
            self.content[uri][key] = item

    def delitem(self, uri: URI, key: str):
        del self.content[uri][key]

    def keys(self, uri: URI) -> Generator[str, None, None]:
        yield from self.content[uri].keys()

    def items(self, uri: URI) -> Generator[tuple[str, Item], None, None]:
        yield from self.content[uri].items()
