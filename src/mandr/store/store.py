from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from mandr.item import DisplayType, Item, ItemMetadata

if TYPE_CHECKING:
    from mandr.storage import URI, Storage


class Store:
    def __init__(self, uri: URI, storage: Storage = None):
        self.uri = uri
        self.storage = storage

    def __eq__(self, other):
        return (
            isinstance(other, Store)
            and (self.uri == other.uri)
            and (self.storage == other.storage)
        )

    def __iter__(self):
        yield from ((key, item.data) for key, item in self.storage.items(self.uri))

    def insert(self, key, value, *, display_type: DisplayType | None = None):
        if self.storage.contains(self.uri, key):
            raise KeyError(key)

        now = datetime.now(tz=UTC).isoformat()
        item = Item(
            data=value,
            metadata=ItemMetadata(
                display_type=(display_type or DisplayType.infer(value)),
                created_at=now,
                updated_at=now,
            ),
        )

        self.storage.setitem(self.uri, key, item)

    def read(self, key):
        if not self.storage.contains(self.uri, key):
            raise KeyError(key)

        return self.storage.getitem(self.uri, key).data

    def update(self, key, value, *, display_type: DisplayType | None = None):
        if not self.storage.contains(self.uri, key):
            raise KeyError(key)

        created_at = self.storage.getitem(self.uri, key).metadata.created_at
        now = datetime.now(tz=UTC).isoformat()
        item = Item(
            data=value,
            metadata=ItemMetadata(
                display_type=(display_type or DisplayType.infer(value)),
                created_at=created_at,
                updated_at=now,
            ),
        )

        self.storage.delitem(self.uri, key)
        self.storage.setitem(self.uri, key, item)

    def delete(self, key):
        if not self.storage.contains(self.uri, key):
            raise KeyError(key)

        return self.storage.delitem(self.uri, key)
