"""
PickleItem.

This module defines the ``PickleItem`` class used to serialize objects that cannot be
otherwise, using binary protocols.
"""

from __future__ import annotations

from functools import cached_property
from io import BytesIO
from typing import Any, TypeVar

from joblib import dump, load

from .item import Item, b64_str_to_bytes, bytes_to_b64_str

# Create a generic variable that can be `PickleItem`, or any subclass.
T = TypeVar("T", bound="PickleItem")


class PickleItem(Item):
    """Serialize objects that cannot be otherwise, using binary protocols."""

    def __init__(self, pickle_b64_str: str):
        """
        Initialize a ``PickleItem``.

        Parameters
        ----------
        pickle_b64_str : str
            The raw bytes of the pickled object encoded in a base64 str.
        """
        self.pickle_b64_str = pickle_b64_str

    @cached_property
    def __raw__(self) -> Any:
        """Get the deserialized python object from the ``PickleItem`` instance."""
        pickle_bytes = b64_str_to_bytes(self.pickle_b64_str)

        with BytesIO(pickle_bytes) as stream:
            return load(stream)

    @property
    def __representation__(self) -> dict:
        """Get the representation of the ``PickleItem`` instance."""
        return {
            "representation": {
                "media_type": "text/markdown",
                "value": f"```python\n{repr(self.__raw__)}\n```",
            }
        }

    @classmethod
    def factory(cls: type[T], value: Any, /) -> T:
        """
        Create a new ``PickleItem`` from ``value`` using binary protocols.

        Parameters
        ----------
        value: Any
            The value to serialize.

        Returns
        -------
        PickleItem
            A new ``PickleItem`` instance.
        """
        with BytesIO() as stream:
            dump(value, stream)

            pickle_bytes = stream.getvalue()
            pickle_b64_str = bytes_to_b64_str(pickle_bytes)

        instance = cls(pickle_b64_str)
        instance.__raw__ = value

        return instance
