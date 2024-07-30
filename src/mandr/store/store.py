"""Object used to store pairs of (key, value) by URI over a storage."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from mandr.item import DisplayType, Item, ItemMetadata
from mandr.storage import URI

if TYPE_CHECKING:
    from pathlib import PosixPath
    from typing import Any, Generator

    from mandr.storage import Storage


class Store:
    """Object used to store pairs of (key, value) by URI over a storage."""

    def __init__(self, uri: URI | PosixPath | str, storage: Storage = None):
        self.uri = URI(uri)
        self.storage = storage

    def __eq__(self, other: Any):
        """Return self == other."""
        return (
            isinstance(other, Store)
            and (self.uri == other.uri)
            and (self.storage == other.storage)
        )

    def __iter__(self) -> Generator[tuple[str, Any], None, None]:
        """Yield the pairs (key, value)."""
        for key, item in self.storage.items():
            if key.parent == self.uri:
                yield (key.stem, item.data)

    def insert(self, key: str, value: Any, *, display_type: DisplayType | None = None):
        """Insert the value for the specified key.

        Parameters
        ----------
        key : str
        value : Any
        display_type : DisplayType, optional
            The type used to display a representation of the value.

        Notes
        -----
        Key will be referenced in the storage in a flat pattern with "u/r/i/keyname".

        Raises
        ------
        KeyError
            If the store already has the specified key.
        """
        key = self.uri / key

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

    def read(self, key: str) -> Any:
        """Return the value for the specified key.

        Raises
        ------
        KeyError
            If the store doesn't have the specified key.
        """
        key = self.uri / key

        if key not in self.storage:
            raise KeyError(key)

        return self.storage.getitem(key).data

    def update(self, key: str, value: Any, *, display_type: DisplayType | None = None):
        """Update the value for the specified key.

        Parameters
        ----------
        key : str
        value : Any
        display_type : DisplayType, optional
            The type used to display a representation of the value.

        Raises
        ------
        KeyError
            If the store doesn't have the specified key.
        """
        key = self.uri / key

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

    def delete(self, key: str):
        """Delete the specified key and its value.

        Raises
        ------
        KeyError
            If the store doesn't have the specified key.
        """
        key = self.uri / key

        if key not in self.storage:
            raise KeyError(key)

        return self.storage.delitem(key)
