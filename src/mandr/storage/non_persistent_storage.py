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

    def __contains__(self, key: URI) -> bool:
        return (key in self.content)

    def getitem(self, key: URI) -> Item:
        return self.content[key]

    def setitem(self, key: URI, item: Item):
        self.content[key] = item

    def delitem(self, key: URI):
        del self.content[key]

    def keys(self) -> Generator[URI, None, None]:
        yield from self.content.keys()

    def items(self) -> Generator[tuple[URI, Item], None, None]:
        yield from self.content.items()
