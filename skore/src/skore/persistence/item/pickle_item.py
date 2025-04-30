"""PickleItem.

This module defines the PickleItem class, which is used to persist objects that cannot
be otherwise.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, Optional

import joblib

from skore.persistence.item.item import Item
from skore.utils import b64_str_to_bytes, bytes_to_b64_str


class PickleItem(Item):
    """
    An item used to persist objects that cannot be otherwise, using binary protocols.

    It encapsulates the object with its pickle representaton, its creation and update
    timestamps.
    """

    def __init__(
        self,
        pickle_b64_str: str,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        note: Optional[str] = None,
    ):
        """
        Initialize a PickleItem.

        Parameters
        ----------
        pickle_b64_str : str
            The raw bytes of the object pickled representation.
        created_at : str, optional
            The creation timestamp in ISO format.
        updated_at : str, optional
            The last update timestamp in ISO format.
        note : str, optional
            A note.
        """
        super().__init__(created_at, updated_at, note)

        self.pickle_b64_str = pickle_b64_str

    @classmethod
    def factory(cls, object: Any, /, **kwargs) -> PickleItem:
        """
        Create a new PickleItem from ``object``.

        Parameters
        ----------
        object: Any
            The object to store.

        Returns
        -------
        PickleItem
            A new PickleItem instance.
        """
        with BytesIO() as stream:
            joblib.dump(object, stream)

            pickle_bytes = stream.getvalue()
            pickle_b64_str = bytes_to_b64_str(pickle_bytes)

            return cls(pickle_b64_str, **kwargs)

    @property
    def object(self) -> Any:
        """The object from the persistence."""
        pickle_bytes = b64_str_to_bytes(self.pickle_b64_str)

        with BytesIO(pickle_bytes) as stream:
            return joblib.load(stream)
