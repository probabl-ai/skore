"""Base class for all items in the project."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional


class ItemTypeError(Exception):
    """Item type exception.

    Exception raised when an attempt is made to convert an object to an Item, but the
    object's type is not supported.
    """


class Item(ABC):
    """
    Abstract base class for all items in the project.

    This class provides a common interface for all items, including
    creation and update timestamps.

    Parameters
    ----------
    created_at : Union[str, None], optional
        The creation timestamp of the item. If None, the current time is used.
    updated_at : Union[str, None], optional
        The last update timestamp of the item. If None, the current time is used.
    note : Union[str, None]
        An optional note.

    Attributes
    ----------
    created_at : str
        The creation timestamp of the item.
    updated_at : str
        The last update timestamp of the item.
    note : Union[str, None]
        An optional note.
    """

    def __init__(
        self,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        note: Optional[str] = None,
    ):
        now = datetime.now(tz=timezone.utc).isoformat()

        self.created_at = created_at or now
        self.updated_at = updated_at or now
        self.note = note

    @classmethod
    @abstractmethod
    def factory(cls, *args, **kwargs) -> Item:
        """
        Create and return a new instance of the Item.

        Returns
        -------
        Item
            A new instance of the Item.
        """

    @property
    def __parameters__(self) -> dict[str, Any]:
        """
        Get the parameters of the Item instance.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the parameters of the Item instance.
        """
        cls = self.__class__
        cls_parameters = inspect.signature(cls).parameters

        return {parameter: getattr(self, parameter) for parameter in cls_parameters}

    def __repr__(self) -> str:
        """Represent the item."""
        return f"{self.__class__.__name__}(...)"
