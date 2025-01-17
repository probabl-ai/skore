"""PickleItem.

This module defines the PickleItem class, which is used to persist objects that cannot
be otherwise.
"""

from __future__ import annotations

from io import BytesIO
from traceback import format_exception
from typing import Any, Optional

import joblib

from .item import Item


class PickleItem(Item):
    """
    An item used to persist objects that cannot be otherwise, using binary protocols.

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
            The raw bytes of the object pickled representation.
        created_at : str, optional
            The creation timestamp in ISO format.
        updated_at : str, optional
            The last update timestamp in ISO format.
        note : str, optional
            A note.
        """
        super().__init__(created_at, updated_at, note)

        self.pickle_bytes = pickle_bytes

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

            return cls(stream.getvalue(), **kwargs)

    @property
    def object(self) -> Any:
        """The object from the persistence."""
        with BytesIO(self.pickle_bytes) as stream:
            return joblib.load(stream)

    def as_serializable_dict(self):
        """Convert item to a JSON-serializable dict to used by frontend."""
        try:
            object = self.object
        except Exception as e:
            value = "Item cannot be displayed"
            traceback = "".join(format_exception(None, e, e.__traceback__))
            note = "".join(
                (
                    (self.note or ""),
                    "\n\n",
                    "UnpicklingError with complete traceback:",
                    "\n\n",
                    traceback,
                )
            )
        else:
            value = f"```python\n{repr(object)}\n```"
            note = self.note

        return super().as_serializable_dict() | {
            "media_type": "text/markdown",
            "value": value,
            "note": note,
        }
