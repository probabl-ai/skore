"""
JSONableItem.

This module defines the ``JSONableItem`` class used to serialize objects using the
``JSON`` format.
"""

from __future__ import annotations

from json import dumps, loads
from typing import Any

from .item import Item, ItemTypeError


class JSONableItem(Item):
    """Serialize objects using the ``JSON`` format."""

    def __init__(self, value: Any):
        """
        Initialize a ``JSONableItem``.

        Parameters
        ----------
        value : Any
            The value.
        """
        self.value = value

    @property
    def __raw__(self):
        """Get the value from the ``JSONableItem`` instance."""
        return self.value

    @property
    def __representation__(self) -> dict:
        """Get the representation of the ``JSONableItem`` instance."""
        return {
            "representation": {
                "media_type": "application/json",
                "value": self.value,
            }
        }

    @classmethod
    def factory(cls, value: Any, /) -> JSONableItem:
        """
        Create a new ``JSONableItem`` from ``value`` using the ``JSON`` format.

        Parameters
        ----------
        value: Any
            The value to serialize.

        Returns
        -------
        JSONableItem
            A new ``JSONableItem`` instance.

        Raises
        ------
        ItemTypeError
            If ``value`` cannot be serialized using the ``JSON`` format.
        """
        try:
            value = loads(dumps(value))
        except TypeError:
            raise ItemTypeError(f"Type '{value.__class__}' is not supported.") from None

        return cls(value)
