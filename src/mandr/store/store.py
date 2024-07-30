from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from mandr.item import DisplayType, Item, ItemMetadata
from mandr.storage import URI

if TYPE_CHECKING:
    from mandr.storage import Storage


class Store:
    def __init__(self, uri: URI | PosixPath | str, storage: Storage = None):
        self.uri = URI(uri)
        self.storage = storage

    def __eq__(self, other):
        return (
            isinstance(other, Store)
            and (self.uri == other.uri)
            and (self.storage == other.storage)
        )

    def __iter__(self):
        for key, item in self.storage.items():
            if key.parent == self.uri:
                yield (key.stem, item.data)

    def insert(self, key, value, *, display_type: DisplayType | None = None):
        key = (self.uri / key)

        if key in self.storage:
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

        self.storage.setitem(key, item)

    def read(self, key):
        key = (self.uri / key)

        if key not in self.storage:
            raise KeyError(key)

        return self.storage.getitem(key).data

    def update(self, key, value, *, display_type: DisplayType | None = None):
        key = (self.uri / key)

        if key not in self.storage:
            raise KeyError(key)

        created_at = self.storage.getitem(key).metadata.created_at
        now = datetime.now(tz=UTC).isoformat()
        item = Item(
            data=value,
            metadata=ItemMetadata(
                display_type=(display_type or DisplayType.infer(value)),
                created_at=created_at,
                updated_at=now,
            ),
        )

        self.storage.delitem(key)
        self.storage.setitem(key, item)

    def delete(self, key):
        key = (self.uri / key)

        if key not in self.storage:
            raise KeyError(key)

        return self.storage.delitem(key)
