"""Object used to store pairs of (key, value) by URI over a storage."""

from __future__ import annotations

import dataclasses
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import RootModel

from mandr.item import DisplayType, Item, ItemMetadata
from mandr.storage import URI, FileSystem
from mandr.store.layout import Layout

if TYPE_CHECKING:
    from pathlib import PosixPath
    from typing import Any, Generator

    from mandr.storage import Storage


def _get_storage_path(MANDR_ROOT: str | None) -> Path:
    """Decide on the `Storage`'s location based on MANDR_ROOT."""
    if MANDR_ROOT is None:
        return Path.cwd() / ".datamander"

    if not Path(MANDR_ROOT).is_absolute():
        return Path.cwd() / MANDR_ROOT

    return Path(MANDR_ROOT)


class Store:
    """Object used to store pairs of (key, value) by URI over a storage."""

    # FIXME find a better to isolate layout from users items
    LAYOUT_KEY = "__mandr__layout__"

    def __init__(self, uri: URI | PosixPath | str, storage: Storage = None):
        self.uri = URI(uri)
        if storage is None:
            directory = _get_storage_path(os.environ.get("MANDR_ROOT"))
            self.storage = FileSystem(directory=directory)
        else:
            self.storage = storage

    def __eq__(self, other: Any):
        """Return self == other."""
        return (
            isinstance(other, Store)
            and (self.uri == other.uri)
            and (self.storage == other.storage)
        )

    def insert(
        self, key: str, value: Any, *, display_type: DisplayType | str | None = None
    ):
        """Insert the value for the specified key.

        Parameters
        ----------
        key : str
        value : Any
        display_type : DisplayType or str, optional
            The type used to display a representation of the value.

        Notes
        -----
        Key will be referenced in the storage in a flat pattern with "u/r/i/keyname".

        Raises
        ------
        KeyError
            If the store already has the specified key.
        """
        uri = self.uri / key

        if uri in self.storage:
            raise KeyError(
                key,
                f"Key '{key}' already exists in {self}; "
                "update or delete the key instead.",
            )

        now = datetime.now(tz=UTC).isoformat()
        display_type = (
            DisplayType(display_type) if display_type else DisplayType.infer(value)
        )
        item = Item(
            data=value,
            metadata=ItemMetadata(
                display_type=display_type,
                created_at=now,
                updated_at=now,
            ),
        )

        self.storage.setitem(uri, item)

    def read(self, key: str, *, metadata: bool = False) -> Any | tuple[Any, dict]:
        """Return the value for the specified key, optionally with its metadata.

        Raises
        ------
        KeyError
            If the store doesn't have the specified key.
        """
        try:
            item = self.storage.getitem(self.uri / key)
        except KeyError as e:
            raise KeyError(key, f"Key '{key}' does not exist in {self}.") from e

        return (
            item.data
            if not metadata
            else (item.data, dataclasses.asdict(item.metadata))
        )

    def update(
        self, key: str, value: Any, *, display_type: DisplayType | str | None = None
    ):
        """Update the value for the specified key.

        Parameters
        ----------
        key : str
        value : Any
        display_type : DisplayType or str, optional
            The type used to display a representation of the value.

        Raises
        ------
        KeyError
            If the store doesn't have the specified key.
        """
        uri = self.uri / key

        if uri not in self.storage:
            raise KeyError(key, f"Key '{key}' does not exist in {self}.")

        created_at = self.storage.getitem(uri).metadata.created_at
        updated_at = datetime.now(tz=UTC).isoformat()
        display_type = (
            DisplayType(display_type) if display_type else DisplayType.infer(value)
        )
        item = Item(
            data=value,
            metadata=ItemMetadata(
                display_type=display_type,
                created_at=created_at,
                updated_at=updated_at,
            ),
        )

        self.storage.delitem(uri)
        self.storage.setitem(uri, item)

    def delete(self, key: str):
        """Delete the specified key and its value.

        Raises
        ------
        KeyError
            If the store doesn't have the specified key.
        """
        try:
            self.storage.delitem(self.uri / key)
        except KeyError as e:
            raise KeyError(key, f"Key '{key}' does not exist in {self}.") from e

    def __iter__(self) -> Generator[str, None, None]:
        """Yield the keys."""
        yield from (key.stem for key in self.storage if key.parent == self.uri)

    def keys(self) -> Generator[str, None, None]:
        """Yield the keys."""
        yield from self

    def items(
        self, *, metadata: bool = False
    ) -> Generator[tuple[str, Any] | tuple[str, Any, dict], None, None]:
        """Yield the pairs(key, value), optionally with the value metadata."""
        for key, item in self.storage.items():
            if key.parent == self.uri:
                yield (
                    (key.stem, item.data)
                    if not metadata
                    else (key.stem, item.data, dataclasses.asdict(item.metadata))
                )

    def get_layout(self) -> Layout:
        """Get the layout, or `[]` if the layout was never set."""
        try:
            layout: Layout = self.read(Store.LAYOUT_KEY)  # type: ignore
        except KeyError:
            layout: Layout = []
        return layout

    def set_layout(self, layout: Layout) -> None:
        """Set the layout to `layout`.

        Raises
        ------
        KeyError
            If `layout` refers to a key which is not in the Store.

        pydantic.ValidationError
            If `layout` is malformed, e.g. if "size" is not a valid
            `LayoutItemSize`.


        Examples
        --------
        >>> Mandr("my_test_root").set_layout([          # doctest: +SKIP
        ...     {"key": "my_integer", "size": "small"},
        ...     {"key": "my_string", "size": "medium"},
        ...     {"key": "my_array", "size": "large"},
        ... ])
        """
        layout = RootModel[Layout].model_validate(layout).root

        for layout_item in layout:
            if layout_item.key not in self.keys():
                raise KeyError(f"Key '{layout_item.key}' is not in the store.")

        try:
            self.insert(Store.LAYOUT_KEY, layout)
        except KeyError:
            self.update(Store.LAYOUT_KEY, layout)
