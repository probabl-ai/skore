"""PickleItem.

This module defines the PickleItem class, which is used to persist objects that cannot
be otherwise.
"""

from __future__ import annotations

from pickle import dumps, loads
from typing import Any, Optional

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
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        note: Optional[str] = None,
    ):
        """
        Initialize a PickleItem.

        Parameters
        ----------
        pickle_bytes : bytes
            The raw bytes of the object pickle representation.
        created_at : str, optional
            The creation timestamp in ISO format.
        updated_at : str, optional
            The last update timestamp in ISO format.
        note : str, optional
            A note.
        """
        super().__init__(created_at, updated_at, note)

        self.pickle_bytes = pickle_bytes

    @property
    def object(self) -> Any:
        """The object from the persistence."""
        return loads(self.pickle_bytes)

    @classmethod
    def factory(cls, object: Any, /, **kwargs) -> PickleItem:
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
        return cls(dumps(object), **kwargs)

    def as_serializable_dict(self):
        """Get a JSON serializable representation of the item."""
        return super().as_serializable_dict() | {
            "media_type": "text/markdown",
            "value": repr(self.object),
        }
