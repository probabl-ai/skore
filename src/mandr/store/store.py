"""Object used to store pairs of (key, value) by URI over a storage."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from mandr.exporter import exporter
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
            raise KeyError(key)

        now = datetime.now(tz=UTC).isoformat()

        if display_type is None:
            display_type = DisplayType.infer(value)

        match display_type:
            case DisplayType.CROSS_VALIDATION_RESULTS:
                if not isinstance(value, dict) or value.get("test_score") is None:
                    raise ValueError(
                        "The input should be of the form output by "
                        "`sklearn.model_selection.cross_validate`"
                    )
                import altair as alt
                import pandas as pd

                cv_results_table = exporter.DataFrame(data=pd.DataFrame(value))
                test_score_plot = exporter.Vega(
                    data=(
                        alt.Chart(pd.DataFrame(pd.DataFrame(value))[["test_score"]])
                        .mark_bar(size=50)
                        .encode(
                            x=alt.X("test_score:Q")
                            .title("Test score")
                            .scale(zero=True),
                            y=(
                                alt.Y("count():Q", axis=alt.Axis(tickMinStep=1)).title(
                                    "Frequency"
                                )
                            ),
                        )
                        .properties(title="Frequency plot of test score")
                    )
                )

                additional_item_metadata = dict(
                    computed=dict(
                        cv_results_table=cv_results_table,
                        test_score_plot=test_score_plot,
                    )
                )
            case _:
                additional_item_metadata = dict()

        item_metadata = dict(
            display_type=display_type,
            created_at=now,
            updated_at=now,
        )

        item = Item(
            data=value,
            metadata=ItemMetadata(**(item_metadata | additional_item_metadata)),
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
            raise KeyError(key) from e

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
            raise KeyError(key)

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
            raise KeyError(key) from e

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
