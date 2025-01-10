"""PickleItem.

This module defines the PickleItem class, which is used to persist objects that cannot
be otherwise.
"""

from __future__ import annotations

from functools import cached_property
from pickle import dumps, loads
from typing import Any

from .item import Item


class PickleItem(Item):
    """
    A class to represent any object item.

    This class is generally used to persist objects that cannot be otherwise.
    It encapsulates the object with its pickle representaton, its creation and update
    timestamps.
    """

    def __init__(
        self,
        pickle_bytes: bytes,
        created_at: str | None = None,
        updated_at: str | None = None,
    ):
        """
        Initialize a PickleItem.

        Parameters
        ----------
        pickle_bytes : bytes
            The raw bytes of the object pickle representation.
        created_at : str
            The creation timestamp in ISO format.
        updated_at : str
            The last update timestamp in ISO format.
        """
        super().__init__(created_at, updated_at)

        self.pickle_bytes = pickle_bytes

    @cached_property
    def object(self) -> Any:
        """The object from the persistence."""
        return loads(self.pickle_bytes)

    @classmethod
    def factory(cls, object: Any) -> PickleItem:
        """
        Create a new PickleItem with any object.

        Parameters
        ----------
        object: Any
            The object to store.

        Returns
        -------
        PickleItem
            A new PickleItem instance.
        """
        return cls(dumps(object))

    def as_serializable_dict(self):
        """Get a JSON serializable representation of the item."""
        return super().as_serializable_dict() | {
            "media_type": "text/markdown",
            "value": repr(self.object),
        }
